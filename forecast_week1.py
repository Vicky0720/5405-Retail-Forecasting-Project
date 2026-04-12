from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing


DEFAULT_API_SERVER = "http://10.246.113.99:3002"
DEFAULT_API_PREFIX = "/api/v1/"
DEFAULT_SCENARIO = "test_world"
DEFAULT_HORIZON = 28
DEFAULT_WBASELINE_WEIGHT = 0.6
DEFAULT_ETR_WEIGHT = 0.4
DEFAULT_LGBM_WEIGHT = 0.0
DEFAULT_TOPDOWN_WEIGHT = 0.0
DEFAULT_BACKTEST_ORIGINS = 10
DEFAULT_HOLDOUT_ORIGINS = 2
WEIGHT_GRID_STEP = 0.1
MODEL_NAMES = ("wbaseline", "etr", "lgbm", "item_topdown")
HORIZON_BUCKETS = ((1, 7), (8, 14), (15, 21), (22, 28))

SALES_VALUE_CANDIDATES = [
    "sales_qty",
    "sales_quantity",
    "quantity",
    "sales_count",
    "sales",
    "qty",
    "gmv",
    "sales_amount",
]


@dataclass
class ForecastArtifacts:
    history: pd.DataFrame
    forecast_panel: pd.DataFrame
    submission_payload: list[dict]
    backtest_summary: pd.DataFrame | None = None
    selected_weights: dict[str, float] | None = None
    selected_bucket_weights: dict[str, dict[str, float]] | None = None
    backtest_predictions_long: pd.DataFrame | None = None
    backtest_metrics_by_horizon: pd.DataFrame | None = None
    backtest_metrics_by_bucket: pd.DataFrame | None = None


def build_api_endpoint(api_server: str = DEFAULT_API_SERVER) -> str:
    return urllib.parse.urljoin(api_server, DEFAULT_API_PREFIX)


def get_from_api(
    api_path: str,
    api_endpoint: str,
    access_token: str | None = None,
    timeout: int = 60,
    **query: object,
) -> pd.DataFrame:
    url = urllib.parse.urljoin(api_endpoint, api_path)
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    headers = {"Authorization": f"Bearer {access_token}"} if access_token else None
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected list response from {url}, got: {type(data).__name__}")
    return pd.DataFrame(data)


def detect_value_column(df: pd.DataFrame) -> str:
    for column in SALES_VALUE_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not identify the sales target column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_history(raw_sales: pd.DataFrame, options_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if "option_id" not in raw_sales.columns:
        raise ValueError("Expected `option_id` in raw sales data.")
    if "date" not in raw_sales.columns:
        raise ValueError("Expected `date` in raw sales data.")

    value_column = detect_value_column(raw_sales)

    history = raw_sales.copy()
    history["date"] = pd.to_datetime(history["date"])
    history[value_column] = pd.to_numeric(history[value_column], errors="coerce").fillna(0.0)

    history = (
        history.groupby(["option_id", "date"], as_index=False)[value_column]
        .sum()
        .rename(columns={value_column: "target"})
        .sort_values(["option_id", "date"])
    )

    option_ids: Iterable[str]
    if options_df is not None and "option_id" in options_df.columns:
        option_ids = sorted(options_df["option_id"].astype(str).unique())
    else:
        option_ids = sorted(history["option_id"].astype(str).unique())

    min_date = history["date"].min()
    max_date = history["date"].max()
    full_index = pd.MultiIndex.from_product(
        [option_ids, pd.date_range(min_date, max_date, freq="D")],
        names=["option_id", "date"],
    )
    history = (
        history.set_index(["option_id", "date"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .sort_values(["option_id", "date"])
    )
    history["target"] = history["target"].astype(float)
    return history


def build_raw_daily_panel(raw_sales: pd.DataFrame) -> pd.DataFrame:
    value_column = detect_value_column(raw_sales)
    daily = raw_sales.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    agg_map: dict[str, str] = {
        value_column: "sum",
    }
    if "sales_amount" in daily.columns:
        agg_map["sales_amount"] = "sum"
    if "label_amount" in daily.columns:
        agg_map["label_amount"] = "sum"
    if "is_online_begin" in daily.columns:
        agg_map["is_online_begin"] = "max"
    if "is_online_end" in daily.columns:
        agg_map["is_online_end"] = "max"
    if "player_group_id" in daily.columns:
        agg_map["player_group_id"] = "last"

    grouped = daily.groupby(["option_id", "date"], as_index=False).agg(agg_map)
    grouped = grouped.rename(columns={value_column: "target"})
    for col in ["sales_amount", "label_amount", "target"]:
        if col in grouped.columns:
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
    for col in ["is_online_begin", "is_online_end"]:
        if col in grouped.columns:
            grouped[col] = grouped[col].fillna(False).astype(int)
    return grouped.sort_values(["option_id", "date"]).reset_index(drop=True)


def build_dow_profile(values: np.ndarray, dates: pd.Series, default_level: float) -> dict[int, float]:
    frame = pd.DataFrame({"target": values}, index=pd.to_datetime(dates))
    frame["dow"] = frame.index.dayofweek
    overall = max(float(frame["target"].mean()), 0.0)

    profile: dict[int, float] = {}
    for dow in range(7):
        group = frame.loc[frame["dow"] == dow, "target"]
        if len(group) == 0:
            profile[dow] = default_level
            continue
        # Shrink weekday means toward the overall level to avoid unstable spikes.
        shrink = min(len(group) / 6.0, 1.0)
        weekday_mean = float(group.tail(8).mean())
        profile[dow] = shrink * weekday_mean + (1.0 - shrink) * max(overall, default_level)
    return profile


def forecast_series_baseline(
    dates: pd.Series,
    target: np.ndarray,
    horizon: int,
) -> np.ndarray:
    dates = pd.to_datetime(dates)
    recent7 = float(np.mean(target[-7:])) if len(target) >= 7 else float(np.mean(target))
    recent28 = float(np.mean(target[-28:])) if len(target) >= 28 else float(np.mean(target))
    recent56 = float(np.mean(target[-56:])) if len(target) >= 56 else recent28

    last14 = float(np.mean(target[-14:])) if len(target) >= 14 else recent7
    prev14 = float(np.mean(target[-28:-14])) if len(target) >= 28 else last14
    raw_trend = (last14 + 1.0) / (prev14 + 1.0)
    trend = float(np.clip(raw_trend, 0.7, 1.35))

    non_zero_share = float(np.mean(target[-56:] > 0)) if len(target) >= 56 else float(np.mean(target > 0))
    level = 0.55 * recent7 + 0.30 * recent28 + 0.15 * recent56
    dow_profile = build_dow_profile(target, dates, default_level=max(level, 0.0))
    overall_recent = max(recent28, 1e-6)

    last_date = dates.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = []
    for step, target_date in enumerate(future_dates, start=1):
        dow = int(target_date.dayofweek)
        weekday_level = dow_profile.get(dow, level)
        seasonal_ratio = np.clip(weekday_level / overall_recent, 0.35, 2.8)
        trend_weight = min(step / 14.0, 1.0)
        damped_trend = 1.0 + (trend - 1.0) * trend_weight
        pred = level * seasonal_ratio * damped_trend

        # Keep sparse options conservative so they do not over-forecast zeros.
        if non_zero_share < 0.25:
            pred *= max(non_zero_share / 0.25, 0.4)
        forecast.append(max(float(pred), 0.0))
    return np.asarray(forecast, dtype=float)


def forecast_one_option(history_one: pd.DataFrame, horizon: int) -> pd.DataFrame:
    history_one = history_one.sort_values("date").reset_index(drop=True)
    option_id = history_one["option_id"].iloc[0]
    target = history_one["target"].to_numpy(dtype=float)
    dates = pd.to_datetime(history_one["date"])

    future_dates = pd.date_range(dates.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = forecast_series_baseline(dates=dates, target=target, horizon=horizon)

    rows: list[dict] = []
    for target_date, pred in zip(future_dates, preds, strict=True):
        rows.append(
            {
                "option_id": option_id,
                "target_date": target_date,
                "forecast": max(float(pred), 0.0),
            }
        )

    return pd.DataFrame(rows)


def run_baseline_forecast(history: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    outputs = []
    for _, history_one in history.groupby("option_id", sort=True):
        outputs.append(forecast_one_option(history_one, horizon=horizon))

    forecast_panel = pd.concat(outputs, ignore_index=True)
    forecast_panel["target_date"] = pd.to_datetime(forecast_panel["target_date"])
    return forecast_panel.sort_values(["option_id", "target_date"]).reset_index(drop=True)


def load_events_from_api(
    api_endpoint: str,
    access_token: str,
    scenario_name: str,
) -> pd.DataFrame:
    events = get_from_api(
        "data/dim/events",
        api_endpoint,
        access_token=access_token,
        scenario_name=scenario_name,
    )
    if events.empty:
        return pd.DataFrame(columns=["date", "event_name", "event_type"])
    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    events["event_name"] = events["name"].fillna("none")
    events["event_type"] = events["type"].fillna("none")
    return events[["date", "event_name", "event_type"]].drop_duplicates()


def weighted_1mape(actual: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    atomic = np.where(
        actual_arr == 0.0,
        0.0,
        1.0 - np.abs(actual_arr - forecast_arr) / np.maximum(np.abs(actual_arr), 1e-12),
    )
    weights = np.maximum(actual_arr, 0.0)
    if float(weights.sum()) == 0.0:
        return 0.0
    return float(np.dot(atomic, weights) / weights.sum())


def horizon_to_bucket(horizon: int) -> str:
    for start, end in HORIZON_BUCKETS:
        if start <= horizon <= end:
            return f"{start}-{end}"
    return f"{horizon}"


def generate_backtest_origins(
    history: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    max_origins: int = DEFAULT_BACKTEST_ORIGINS,
) -> list[pd.Timestamp]:
    max_date = pd.to_datetime(history["date"]).max()
    min_origin = pd.to_datetime(history["date"]).min() + pd.Timedelta(days=56)
    latest_origin = max_date - pd.Timedelta(days=horizon)
    origins: list[pd.Timestamp] = []
    for offset in range(max_origins):
        origin = latest_origin - pd.Timedelta(days=7 * offset)
        if origin >= min_origin:
            origins.append(origin)
    return list(reversed(origins))


def prepare_model_panel(
    history: pd.DataFrame,
    raw_daily: pd.DataFrame,
    options_df: pd.DataFrame,
    items_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    panel = history.merge(options_df, on="option_id", how="left")
    if not raw_daily.empty:
        keep_cols = [
            col
            for col in ["option_id", "date", "sales_amount", "label_amount", "is_online_begin", "is_online_end", "player_group_id"]
            if col in raw_daily.columns
        ]
        panel = panel.merge(raw_daily[keep_cols], on=["option_id", "date"], how="left")
    if not items_df.empty:
        panel = panel.merge(items_df[["item_id", "cate1_id", "cate2_id"]].drop_duplicates(), on="item_id", how="left")

    if not events_df.empty:
        panel = panel.merge(events_df, on="date", how="left")
    else:
        panel["event_name"] = "none"
    panel["event_name"] = panel["event_name"].fillna("none")

    panel["dow"] = panel["date"].dt.dayofweek
    panel["dom"] = panel["date"].dt.day
    panel["month_num"] = panel["date"].dt.month
    panel["weekofyear"] = panel["date"].dt.isocalendar().week.astype(int)
    panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
    panel["trend_idx"] = (panel["date"] - panel["date"].min()).dt.days
    panel["event_type"] = panel["event_type"].fillna("none") if "event_type" in panel.columns else "none"
    panel["player_group_id"] = panel["player_group_id"].fillna("unknown") if "player_group_id" in panel.columns else "unknown"

    for col in ["sales_amount", "label_amount"]:
        if col not in panel.columns:
            panel[col] = 0.0
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    for col in ["is_online_begin", "is_online_end"]:
        if col not in panel.columns:
            panel[col] = 0
        panel[col] = panel[col].fillna(0).astype(int)

    panel["unit_sales_price"] = np.where(panel["target"] > 0, panel["sales_amount"] / panel["target"], 0.0)
    panel["unit_label_price"] = np.where(panel["target"] > 0, panel["label_amount"] / panel["target"], panel["label_price"].fillna(0.0))
    panel["discount_ratio"] = np.where(panel["label_amount"] > 0, panel["sales_amount"] / np.maximum(panel["label_amount"], 1e-6), 1.0)
    panel["discount_depth"] = np.maximum(0.0, 1.0 - panel["discount_ratio"])
    panel["price_gap_to_label"] = panel["label_price"].fillna(0.0) - panel["unit_sales_price"]

    price_ref = options_df.groupby("item_id")["label_price"].median().rename("item_median_label_price")
    panel = panel.merge(price_ref, on="item_id", how="left")
    panel["relative_price_in_item"] = np.where(
        panel["item_median_label_price"].fillna(0.0) > 0,
        panel["label_price"].fillna(0.0) / np.maximum(panel["item_median_label_price"], 1e-6),
        1.0,
    )
    price_rank = (
        options_df[["option_id", "item_id", "label_price"]]
        .assign(price_rank_in_item=lambda df: df.groupby("item_id")["label_price"].rank(method="average", pct=True))
        [["option_id", "price_rank_in_item"]]
    )
    panel = panel.merge(price_rank, on="option_id", how="left")
    panel["price_rank_in_item"] = panel["price_rank_in_item"].fillna(0.5)

    panel["online_state"] = (
        panel.groupby("option_id")["is_online_begin"].cumsum() - panel.groupby("option_id")["is_online_end"].cumsum()
    ).clip(lower=0, upper=1)
    panel["online_state_change"] = (panel["online_state"] != panel.groupby("option_id")["online_state"].shift(1)).astype(int)
    panel["days_since_online_begin"] = (
        panel["date"] - panel["date"].where(panel["is_online_begin"] == 1).groupby(panel["option_id"]).ffill()
    ).dt.days.fillna(999).clip(upper=999)

    item_daily = (
        panel.groupby(["item_id", "date"], as_index=False)["target"]
        .sum()
        .rename(columns={"target": "item_target"})
    )
    panel = panel.merge(item_daily, on=["item_id", "date"], how="left")
    panel["share"] = np.where(panel["item_target"] > 0, panel["target"] / panel["item_target"], 0.0)
    item_promo = (
        panel.groupby(["item_id", "date"], as_index=False)["discount_depth"]
        .mean()
        .rename(columns={"discount_depth": "item_promo_intensity"})
    )
    panel = panel.merge(item_promo, on=["item_id", "date"], how="left")

    for lag in [1, 7, 14, 21, 28, 35, 42, 56]:
        panel[f"lag_{lag}"] = panel.groupby("option_id")["target"].shift(lag)

    for window in [7, 14, 28, 56]:
        shifted = panel.groupby("option_id")["target"].shift(1)
        panel[f"roll_mean_{window}"] = (
            shifted.rolling(window).mean().reset_index(level=0, drop=True)
        )
        panel[f"roll_zero_{window}"] = (
            shifted.rolling(window)
            .apply(lambda x: float(np.mean(np.asarray(x) == 0)), raw=False)
            .reset_index(level=0, drop=True)
        )
        panel[f"discount_depth_mean_{window}"] = (
            panel.groupby("option_id")["discount_depth"].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        )
        panel[f"unit_sales_price_mean_{window}"] = (
            panel.groupby("option_id")["unit_sales_price"].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        )
        panel[f"online_change_count_{window}"] = (
            panel.groupby("option_id")["online_state_change"].shift(1).rolling(window).sum().reset_index(level=0, drop=True)
        )

    for lag in [1, 7, 14, 28]:
        panel[f"item_lag_{lag}"] = panel.groupby("item_id")["item_target"].shift(lag)
        panel[f"discount_depth_lag_{lag}"] = panel.groupby("option_id")["discount_depth"].shift(lag)
        panel[f"unit_sales_price_lag_{lag}"] = panel.groupby("option_id")["unit_sales_price"].shift(lag)

    for window in [7, 28]:
        item_shifted = panel.groupby("item_id")["item_target"].shift(1)
        share_shifted = panel.groupby("option_id")["share"].shift(1)
        panel[f"item_roll_mean_{window}"] = (
            item_shifted.rolling(window).mean().reset_index(level=0, drop=True)
        )
        panel[f"share_roll_mean_{window}"] = (
            share_shifted.rolling(window).mean().reset_index(level=0, drop=True)
        )
        panel[f"item_promo_intensity_mean_{window}"] = (
            panel.groupby("item_id")["item_promo_intensity"].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        )

    panel["vol56"] = panel.groupby("option_id")["target"].shift(1).rolling(56).sum().reset_index(level=0, drop=True)
    panel["nz56"] = (
        panel.groupby("option_id")["target"].shift(1).rolling(56).apply(lambda x: float(np.mean(np.asarray(x) > 0)), raw=False).reset_index(level=0, drop=True)
    )
    panel["vol56_median_by_date"] = panel.groupby("date")["vol56"].transform("median")
    panel["vol56_q25_by_date"] = panel.groupby("date")["vol56"].transform(lambda s: s.quantile(0.25))
    panel["segment"] = "mid"
    panel.loc[(panel["vol56"] >= panel["vol56_median_by_date"]) & (panel["nz56"] >= 0.45), "segment"] = "high_volume"
    panel.loc[(panel["vol56"] <= panel["vol56_q25_by_date"]) | (panel["nz56"] <= 0.20), "segment"] = "sparse"

    return panel.sort_values(["option_id", "date"]).reset_index(drop=True)


def build_direct_etr_features() -> tuple[list[str], list[str]]:
    cat_features = ["option_id", "item_id", "cate1_id", "cate2_id", "event_name", "event_type", "player_group_id", "segment"]
    num_features = [
        "dow",
        "dom",
        "month_num",
        "weekofyear",
        "is_weekend",
        "trend_idx",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "lag_35",
        "lag_42",
        "lag_56",
        "roll_mean_7",
        "roll_mean_14",
        "roll_mean_28",
        "roll_mean_56",
        "roll_zero_7",
        "roll_zero_14",
        "roll_zero_28",
        "roll_zero_56",
        "item_lag_1",
        "item_lag_7",
        "item_lag_14",
        "item_lag_28",
        "item_roll_mean_7",
        "item_roll_mean_28",
        "share_roll_mean_7",
        "share_roll_mean_28",
        "label_price",
        "relative_price_in_item",
        "price_rank_in_item",
        "unit_sales_price_lag_1",
        "unit_sales_price_lag_7",
        "unit_sales_price_lag_14",
        "unit_sales_price_lag_28",
        "discount_depth_lag_1",
        "discount_depth_lag_7",
        "discount_depth_lag_14",
        "discount_depth_lag_28",
        "discount_depth_mean_7",
        "discount_depth_mean_14",
        "discount_depth_mean_28",
        "discount_depth_mean_56",
        "unit_sales_price_mean_7",
        "unit_sales_price_mean_14",
        "unit_sales_price_mean_28",
        "unit_sales_price_mean_56",
        "online_state",
        "days_since_online_begin",
        "online_change_count_7",
        "online_change_count_14",
        "online_change_count_28",
        "online_change_count_56",
        "item_promo_intensity",
        "item_promo_intensity_mean_7",
        "item_promo_intensity_mean_28",
        "vol56",
        "nz56",
    ]
    return cat_features, num_features


def build_preprocessor(cat_features: list[str], num_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_features,
            ),
        ]
    )


def build_etr_model(segment: str = "mid") -> ExtraTreesRegressor:
    if segment == "high_volume":
        n_estimators = 280
        min_samples_leaf = 3
    elif segment == "sparse":
        n_estimators = 240
        min_samples_leaf = 6
    else:
        n_estimators = 260
        min_samples_leaf = 4
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=42,
        min_samples_leaf=min_samples_leaf,
        max_features=0.5,
        n_jobs=-1,
    )


def build_lgbm_model(segment: str = "mid") -> LGBMRegressor:
    if segment == "high_volume":
        num_leaves = 23
        min_child_samples = 40
    elif segment == "sparse":
        num_leaves = 15
        min_child_samples = 60
    else:
        num_leaves = 19
        min_child_samples = 50
    return LGBMRegressor(
        objective="poisson",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        max_depth=6,
        reg_lambda=3.0,
        reg_alpha=0.25,
        subsample=0.8,
        colsample_bytree=0.75,
        bagging_fraction=0.8,
        bagging_freq=1,
        random_state=42,
        verbosity=-1,
    )


def run_direct_model_forecast(
    panel: pd.DataFrame,
    feature_date: pd.Timestamp,
    model_name: str,
    model_builder,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    cat_features, num_features = build_direct_etr_features()
    min_train_date = panel["date"].min() + pd.Timedelta(days=56)
    base_train = panel[(panel["date"] >= min_train_date) & (panel["date"] <= feature_date - pd.Timedelta(days=1))].copy()
    feature_rows = panel.loc[panel["date"] == feature_date, ["option_id"] + cat_features[1:] + num_features].copy()

    if feature_rows.empty:
        raise ValueError(f"No feature rows found for feature_date={feature_date.date()}.")
    if base_train.empty:
        return pd.DataFrame(columns=["option_id", "target_date", "forecast"])

    preprocessor = build_preprocessor(cat_features=cat_features, num_features=num_features)

    forecast_parts: list[pd.DataFrame] = []
    model_input_columns = cat_features + num_features

    for step in range(1, horizon + 1):
        train = base_train.copy()
        train["target_h"] = train.groupby("option_id")["target"].shift(-step)
        train = train.dropna(subset=["target_h"])
        segment_predictions: list[pd.DataFrame] = []
        for segment_name in ["high_volume", "mid", "sparse"]:
            train_seg = train[train["segment"] == segment_name]
            test_seg = feature_rows[feature_rows["segment"] == segment_name]
            if test_seg.empty:
                continue
            if len(train_seg) < 100:
                train_seg = train
            model = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model_builder(segment_name)),
                ]
            )
            model.fit(train_seg[model_input_columns], train_seg["target_h"])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning,
                )
                pred = np.clip(model.predict(test_seg[model_input_columns]), 0.0, None)
            segment_predictions.append(
                pd.DataFrame(
                    {
                        "option_id": test_seg["option_id"].to_numpy(),
                        "target_date": feature_date + pd.Timedelta(days=step),
                        "forecast": pred,
                        "model_name": model_name,
                        "segment": segment_name,
                    }
                )
            )
        forecast_parts.append(pd.concat(segment_predictions, ignore_index=True))

    return pd.concat(forecast_parts, ignore_index=True).sort_values(["option_id", "target_date"])


def run_direct_etr_forecast(
    panel: pd.DataFrame,
    feature_date: pd.Timestamp,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    return run_direct_model_forecast(
        panel=panel,
        feature_date=feature_date,
        model_name="etr",
        model_builder=build_etr_model,
        horizon=horizon,
    )[["option_id", "target_date", "forecast"]]


def run_direct_lgbm_forecast(
    panel: pd.DataFrame,
    feature_date: pd.Timestamp,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    return run_direct_model_forecast(
        panel=panel,
        feature_date=feature_date,
        model_name="lgbm",
        model_builder=build_lgbm_model,
        horizon=horizon,
    )[["option_id", "target_date", "forecast"]]


def forecast_series_ets(dates: pd.Series, target: np.ndarray, horizon: int) -> np.ndarray:
    if len(target) < 28 or float(np.sum(target)) == 0.0:
        return forecast_series_baseline(dates=dates, target=target, horizon=horizon)

    try:
        model = ExponentialSmoothing(
            target,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, use_brute=False)
        pred = np.asarray(fitted.forecast(horizon), dtype=float)
        return np.clip(pred, 0.0, None)
    except Exception:
        return forecast_series_baseline(dates=dates, target=target, horizon=horizon)


def compute_recent_option_shares(
    history: pd.DataFrame,
    options_df: pd.DataFrame,
    feature_date: pd.Timestamp,
) -> pd.DataFrame:
    enriched = history.merge(options_df[["option_id", "item_id"]], on="option_id", how="left")

    def aggregate_share(window: int) -> pd.Series:
        start_date = feature_date - pd.Timedelta(days=window - 1)
        subset = enriched[(enriched["date"] >= start_date) & (enriched["date"] <= feature_date)]
        grouped = subset.groupby(["item_id", "option_id"], as_index=False)["target"].sum()
        grouped["window"] = window
        return grouped

    recent = aggregate_share(28).rename(columns={"target": "target_28"})
    long = aggregate_share(84).rename(columns={"target": "target_84"})
    share = recent.merge(long, on=["item_id", "option_id"], how="outer").fillna(0.0)
    share["raw_share"] = 0.7 * share["target_28"] + 0.3 * share["target_84"] + 1e-6
    share["share"] = share["raw_share"] / share.groupby("item_id")["raw_share"].transform("sum")
    return share[["item_id", "option_id", "share"]]


def run_item_topdown_forecast(
    history: pd.DataFrame,
    options_df: pd.DataFrame,
    feature_date: pd.Timestamp,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    item_daily = (
        history.merge(options_df[["option_id", "item_id"]], on="option_id", how="left")
        .groupby(["item_id", "date"], as_index=False)["target"]
        .sum()
        .sort_values(["item_id", "date"])
    )
    shares = compute_recent_option_shares(history=history, options_df=options_df, feature_date=feature_date)

    item_forecasts: list[pd.DataFrame] = []
    for item_id, item_hist in item_daily.groupby("item_id", sort=True):
        dates = pd.to_datetime(item_hist["date"])
        target = item_hist["target"].to_numpy(dtype=float)
        baseline_pred = forecast_series_baseline(dates=dates, target=target, horizon=horizon)
        ets_pred = forecast_series_ets(dates=dates, target=target, horizon=horizon)
        item_pred = 0.5 * baseline_pred + 0.5 * ets_pred
        item_forecasts.append(
            pd.DataFrame(
                {
                    "item_id": item_id,
                    "target_date": pd.date_range(feature_date + pd.Timedelta(days=1), periods=horizon, freq="D"),
                    "item_forecast": item_pred,
                }
            )
        )

    item_fcst = pd.concat(item_forecasts, ignore_index=True)
    option_fcst = item_fcst.merge(shares, on="item_id", how="left")
    option_fcst["forecast"] = option_fcst["item_forecast"] * option_fcst["share"]
    return option_fcst[["option_id", "target_date", "forecast"]].sort_values(["option_id", "target_date"])


def build_candidate_forecasts(
    history: pd.DataFrame,
    raw_daily: pd.DataFrame,
    options_df: pd.DataFrame,
    items_df: pd.DataFrame,
    events_df: pd.DataFrame,
    feature_date: pd.Timestamp,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, pd.DataFrame]:
    history_cut = history[history["date"] <= feature_date].copy()
    raw_cut = raw_daily[raw_daily["date"] <= feature_date].copy() if not raw_daily.empty else raw_daily
    panel = prepare_model_panel(history_cut, raw_daily=raw_cut, options_df=options_df, items_df=items_df, events_df=events_df)
    return {
        "wbaseline": run_baseline_forecast(history_cut, horizon=horizon),
        "etr": run_direct_etr_forecast(panel, feature_date=feature_date, horizon=horizon),
        "lgbm": run_direct_lgbm_forecast(panel, feature_date=feature_date, horizon=horizon),
        "item_topdown": run_item_topdown_forecast(
            history_cut,
            options_df=options_df,
            feature_date=feature_date,
            horizon=horizon,
        ),
    }


def merge_candidate_predictions(
    actual: pd.DataFrame,
    candidate_forecasts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = actual.copy()
    for model_name, frame in candidate_forecasts.items():
        if frame.empty:
            merged[f"forecast_{model_name}"] = np.nan
            continue
        renamed = frame.rename(columns={"forecast": f"forecast_{model_name}"})
        merged = merged.merge(renamed, on=["option_id", "target_date"], how="left")
    return merged


def search_best_ensemble_weights(
    backtest_panel: pd.DataFrame,
    step: float = WEIGHT_GRID_STEP,
) -> tuple[dict[str, float], float]:
    units = int(round(1 / step))
    best_weights = {
        "wbaseline": DEFAULT_WBASELINE_WEIGHT,
        "etr": DEFAULT_ETR_WEIGHT,
        "lgbm": DEFAULT_LGBM_WEIGHT,
        "item_topdown": DEFAULT_TOPDOWN_WEIGHT,
    }
    best_score = -float("inf")

    for combo in product(range(units + 1), repeat=len(MODEL_NAMES)):
        if sum(combo) != units:
            continue
        weights = {name: part / units for name, part in zip(MODEL_NAMES, combo, strict=True)}
        ensemble_pred = sum(
            weights[name] * backtest_panel[f"forecast_{name}"].to_numpy(dtype=float)
            for name in MODEL_NAMES
        )
        score = weighted_1mape(backtest_panel["actual"], ensemble_pred)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, float(best_score)


def search_best_bucket_weights(
    selection_panel: pd.DataFrame,
    step: float = WEIGHT_GRID_STEP,
) -> dict[str, dict[str, float]]:
    weights_by_bucket: dict[str, dict[str, float]] = {}
    for bucket, frame in selection_panel.groupby("bucket"):
        best_weights, _ = search_best_ensemble_weights(frame, step=step)
        weights_by_bucket[str(bucket)] = best_weights
    return weights_by_bucket


def build_long_backtest_predictions(backtest_panel: pd.DataFrame) -> pd.DataFrame:
    long_parts: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        col = f"forecast_{model_name}"
        if col not in backtest_panel.columns:
            continue
        part = backtest_panel[
            ["origin", "split", "option_id", "target_date", "actual", "horizon", "bucket", col]
        ].rename(columns={col: "forecast"})
        part["model"] = model_name
        long_parts.append(part)
    return pd.concat(long_parts, ignore_index=True)


def summarize_backtest_scores(long_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _append_split_rollups(frame: pd.DataFrame, extra_group_cols: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame
        group_cols = ["split", "model", *extra_group_cols]
        rolled = (
            long_panel.groupby(group_cols, as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "score": weighted_1mape(g["actual"], g["forecast"]),
                        "n_rows": len(g),
                        "actual_sum": float(g["actual"].sum()),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        rolled["origin"] = rolled["split"].map(
            {
                "selection": "ALL_SELECTION",
                "holdout": "ALL_HOLDOUT",
            }
        ).fillna("ALL_UNKNOWN")
        ordered_cols = ["origin", "split", "model", *extra_group_cols, "score", "n_rows", "actual_sum"]
        return pd.concat([frame, rolled[ordered_cols]], ignore_index=True)

    overall = (
        long_panel.groupby(["origin", "split", "model"], as_index=False)
        .apply(lambda g: pd.Series({"score": weighted_1mape(g["actual"], g["forecast"]), "n_rows": len(g), "actual_sum": float(g["actual"].sum())}), include_groups=False)
        .reset_index(drop=True)
    )
    by_horizon = (
        long_panel.groupby(["origin", "split", "model", "horizon"], as_index=False)
        .apply(lambda g: pd.Series({"score": weighted_1mape(g["actual"], g["forecast"]), "n_rows": len(g), "actual_sum": float(g["actual"].sum())}), include_groups=False)
        .reset_index(drop=True)
    )
    by_bucket = (
        long_panel.groupby(["origin", "split", "model", "bucket"], as_index=False)
        .apply(lambda g: pd.Series({"score": weighted_1mape(g["actual"], g["forecast"]), "n_rows": len(g), "actual_sum": float(g["actual"].sum())}), include_groups=False)
        .reset_index(drop=True)
    )
    overall = _append_split_rollups(overall, [])
    by_horizon = _append_split_rollups(by_horizon, ["horizon"])
    by_bucket = _append_split_rollups(by_bucket, ["bucket"])
    return overall, by_horizon, by_bucket


def split_backtest_origins(
    origins: list[pd.Timestamp],
    holdout_origins: int = DEFAULT_HOLDOUT_ORIGINS,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    if not origins:
        return [], []
    holdout_count = min(max(holdout_origins, 0), max(len(origins) - 1, 0))
    if holdout_count == 0:
        return origins, []
    return origins[:-holdout_count], origins[-holdout_count:]


def run_backtest(
    history: pd.DataFrame,
    raw_daily: pd.DataFrame,
    options_df: pd.DataFrame,
    items_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    max_origins: int = DEFAULT_BACKTEST_ORIGINS,
    holdout_origins: int = DEFAULT_HOLDOUT_ORIGINS,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origins = generate_backtest_origins(history=history, horizon=horizon, max_origins=max_origins)
    if not origins:
        default_weights = {
            "wbaseline": DEFAULT_WBASELINE_WEIGHT,
            "etr": DEFAULT_ETR_WEIGHT,
            "lgbm": DEFAULT_LGBM_WEIGHT,
            "item_topdown": DEFAULT_TOPDOWN_WEIGHT,
        }
        return pd.DataFrame(), default_weights, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selection_origins, holdout_origin_list = split_backtest_origins(
        origins=origins,
        holdout_origins=holdout_origins,
    )
    if not selection_origins:
        selection_origins = origins
        holdout_origin_list = []
    selection_origin_set = {pd.Timestamp(x) for x in selection_origins}
    holdout_origin_set = {pd.Timestamp(x) for x in holdout_origin_list}

    score_rows: list[dict] = []
    backtest_panels: list[pd.DataFrame] = []

    for origin in origins:
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        actual = (
            history[history["date"].isin(future)][["option_id", "date", "target"]]
            .rename(columns={"date": "target_date", "target": "actual"})
            .sort_values(["option_id", "target_date"])
        )
        candidate_forecasts = build_candidate_forecasts(
            history=history,
            raw_daily=raw_daily,
            options_df=options_df,
            items_df=items_df,
            events_df=events_df,
            feature_date=origin,
            horizon=horizon,
        )
        merged = merge_candidate_predictions(actual, candidate_forecasts)
        merged["origin"] = origin
        merged["split"] = "holdout" if origin in holdout_origin_set else "selection"
        merged["horizon"] = (pd.to_datetime(merged["target_date"]) - pd.to_datetime(origin)).dt.days.astype(int)
        merged["bucket"] = merged["horizon"].map(horizon_to_bucket)
        backtest_panels.append(merged)

        for model_name in MODEL_NAMES:
            score_rows.append(
                {
                    "origin": origin.date().isoformat(),
                    "split": "holdout" if origin in holdout_origin_set else "selection",
                    "model": model_name,
                    "score": weighted_1mape(merged["actual"], merged[f"forecast_{model_name}"]),
                }
            )

    backtest_panel = pd.concat(backtest_panels, ignore_index=True)
    selection_panel = backtest_panel[backtest_panel["split"] == "selection"].copy()
    selected_weights, best_score = search_best_ensemble_weights(selection_panel)
    selected_bucket_weights = search_best_bucket_weights(selection_panel) if not selection_panel.empty else {}

    for origin, frame in backtest_panel.groupby("origin"):
        ensemble_pred = sum(
            selected_weights[name] * frame[f"forecast_{name}"].to_numpy(dtype=float)
            for name in MODEL_NAMES
        )
        score_rows.append(
            {
                "origin": pd.to_datetime(origin).date().isoformat(),
                "split": frame["split"].iloc[0],
                "model": "ensemble_selected",
                "score": weighted_1mape(frame["actual"], ensemble_pred),
                }
            )

    holdout_panel = backtest_panel[backtest_panel["split"] == "holdout"].copy()
    score_rows.append({"origin": "ALL_SELECTION", "split": "selection", "model": "ensemble_selected", "score": best_score})
    if not holdout_origin_set:
        holdout_score = None
    else:
        holdout_pred = sum(
            selected_weights[name] * holdout_panel[f"forecast_{name}"].to_numpy(dtype=float)
            for name in MODEL_NAMES
        )
        holdout_score = weighted_1mape(holdout_panel["actual"], holdout_pred)
        score_rows.append(
            {
                "origin": "ALL_HOLDOUT",
                "split": "holdout",
                "model": "ensemble_selected",
                "score": holdout_score,
            }
        )
    for model_name in MODEL_NAMES:
        score_rows.append(
            {
                "origin": "ALL_SELECTION",
                "split": "selection",
                "model": model_name,
                "score": weighted_1mape(selection_panel["actual"], selection_panel[f"forecast_{model_name}"]),
            }
        )
        if holdout_origin_set:
            score_rows.append(
                {
                    "origin": "ALL_HOLDOUT",
                    "split": "holdout",
                    "model": model_name,
                    "score": weighted_1mape(holdout_panel["actual"], holdout_panel[f"forecast_{model_name}"]),
                }
            )

    long_panel = build_long_backtest_predictions(backtest_panel)
    ensemble_long = backtest_panel[["origin", "split", "option_id", "target_date", "actual", "horizon", "bucket"]].copy()
    ensemble_long["forecast"] = sum(
        selected_weights[name] * backtest_panel[f"forecast_{name}"].to_numpy(dtype=float)
        for name in MODEL_NAMES
    )
    ensemble_long["model"] = "ensemble_selected"
    long_panel = pd.concat([long_panel, ensemble_long], ignore_index=True)
    if selected_bucket_weights:
        bucket_parts: list[pd.DataFrame] = []
        for bucket, frame in backtest_panel.groupby("bucket"):
            weights = selected_bucket_weights.get(str(bucket), selected_weights)
            tmp = frame[["origin", "split", "option_id", "target_date", "actual", "horizon", "bucket"]].copy()
            tmp["forecast"] = sum(
                weights[name] * frame[f"forecast_{name}"].to_numpy(dtype=float)
                for name in MODEL_NAMES
            )
            tmp["model"] = "ensemble_bucket_selected"
            bucket_parts.append(tmp)
        long_panel = pd.concat([long_panel, pd.concat(bucket_parts, ignore_index=True)], ignore_index=True)
    overall_metrics, horizon_metrics, bucket_metrics = summarize_backtest_scores(long_panel)

    return overall_metrics, selected_weights, selected_bucket_weights, long_panel, horizon_metrics, bucket_metrics


def run_ensemble_forecast(
    history: pd.DataFrame,
    raw_daily: pd.DataFrame,
    options_df: pd.DataFrame,
    items_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    wbaseline_weight: float = DEFAULT_WBASELINE_WEIGHT,
    etr_weight: float = DEFAULT_ETR_WEIGHT,
    lgbm_weight: float = DEFAULT_LGBM_WEIGHT,
    topdown_weight: float = DEFAULT_TOPDOWN_WEIGHT,
    bucket_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    feature_date = pd.to_datetime(history["date"]).max()
    candidate_forecasts = build_candidate_forecasts(
        history=history,
        raw_daily=raw_daily,
        options_df=options_df,
        items_df=items_df,
        events_df=events_df,
        feature_date=feature_date,
        horizon=horizon,
    )
    available_weights = {
        "wbaseline": wbaseline_weight,
        "etr": etr_weight,
        "lgbm": lgbm_weight,
        "item_topdown": topdown_weight,
    }
    ensemble = None
    for model_name, frame in candidate_forecasts.items():
        if frame.empty:
            available_weights[model_name] = 0.0
            continue
        renamed = frame.rename(columns={"forecast": f"forecast_{model_name}"})
        if ensemble is None:
            ensemble = renamed
        else:
            ensemble = ensemble.merge(renamed, on=["option_id", "target_date"], how="inner")
    if ensemble is None:
        raise ValueError("No candidate forecasts were produced.")

    for model_name in MODEL_NAMES:
        col = f"forecast_{model_name}"
        if col not in ensemble.columns:
            ensemble[col] = 0.0
    ensemble["horizon"] = (pd.to_datetime(ensemble["target_date"]) - feature_date).dt.days.astype(int)
    ensemble["bucket"] = ensemble["horizon"].map(horizon_to_bucket)

    if bucket_weights:
        preds = []
        for bucket, frame in ensemble.groupby("bucket"):
            weights = bucket_weights.get(str(bucket), available_weights)
            weight_sum = sum(weights.values())
            if weight_sum <= 0:
                weights = {"wbaseline": 1.0, "etr": 0.0, "lgbm": 0.0, "item_topdown": 0.0}
                weight_sum = 1.0
            weights = {name: value / weight_sum for name, value in weights.items()}
            pred = (
                weights["wbaseline"] * frame["forecast_wbaseline"]
                + weights["etr"] * frame["forecast_etr"]
                + weights["lgbm"] * frame["forecast_lgbm"]
                + weights["item_topdown"] * frame["forecast_item_topdown"]
            )
            preds.append(pred)
        ensemble["forecast"] = pd.concat(preds).sort_index()
        return ensemble[["option_id", "target_date", "forecast"]].sort_values(["option_id", "target_date"])

    weight_sum = sum(available_weights.values())
    if weight_sum <= 0:
        available_weights["wbaseline"] = 1.0
        weight_sum = 1.0
    available_weights = {name: value / weight_sum for name, value in available_weights.items()}
    ensemble["forecast"] = (
        available_weights["wbaseline"] * ensemble["forecast_wbaseline"]
        + available_weights["etr"] * ensemble["forecast_etr"]
        + available_weights["lgbm"] * ensemble["forecast_lgbm"]
        + available_weights["item_topdown"] * ensemble["forecast_item_topdown"]
    )
    return ensemble[["option_id", "target_date", "forecast"]].sort_values(["option_id", "target_date"])


def format_submission_payload(
    forecast_panel: pd.DataFrame,
    template: list[dict] | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> list[dict]:
    grouped = (
        forecast_panel.sort_values(["option_id", "target_date"])
        .groupby("option_id")["forecast"]
        .apply(lambda s: [round(float(x), 6) for x in s.tolist()])
        .to_dict()
    )

    if template is None:
        option_ids = sorted(grouped)
        template = [{"option_id": option_id, "forecast": [0.0] * horizon} for option_id in option_ids]

    payload = []
    for row in template:
        option_id = str(row["option_id"])
        values = grouped.get(option_id, [0.0] * horizon)
        if len(values) != horizon:
            raise ValueError(f"Option {option_id} has {len(values)} forecast steps, expected {horizon}.")
        payload.append({"option_id": option_id, "forecast": values})
    return payload


def save_outputs(
    artifacts: ForecastArtifacts,
    output_dir: Path,
    feature_date: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_tag = feature_date.replace("-", "")

    history_path = output_dir / f"history_option_day_{feature_tag}.csv"
    forecast_path = output_dir / f"forecast_option_day_{feature_tag}.csv"
    payload_path = output_dir / f"forecast_submission_{feature_tag}.json"
    backtest_path = output_dir / f"backtest_scores_{feature_tag}.csv"
    weights_path = output_dir / f"selected_weights_{feature_tag}.json"
    bucket_weights_path = output_dir / f"selected_weights_by_bucket_{feature_tag}.json"
    long_backtest_path = output_dir / f"backtest_predictions_long_{feature_tag}.csv"
    horizon_metrics_path = output_dir / f"backtest_metrics_by_horizon_{feature_tag}.csv"
    bucket_metrics_path = output_dir / f"backtest_metrics_by_bucket_{feature_tag}.csv"

    artifacts.history.to_csv(history_path, index=False)
    artifacts.forecast_panel.assign(
        target_date=artifacts.forecast_panel["target_date"].dt.strftime("%Y-%m-%d")
    ).to_csv(forecast_path, index=False)
    payload_path.write_text(json.dumps(artifacts.submission_payload, indent=2), encoding="utf-8")
    if artifacts.backtest_summary is not None and not artifacts.backtest_summary.empty:
        artifacts.backtest_summary.to_csv(backtest_path, index=False)
    if artifacts.selected_weights is not None:
        weights_path.write_text(json.dumps(artifacts.selected_weights, indent=2), encoding="utf-8")
    if artifacts.selected_bucket_weights is not None:
        bucket_weights_path.write_text(json.dumps(artifacts.selected_bucket_weights, indent=2), encoding="utf-8")
    if artifacts.backtest_predictions_long is not None and not artifacts.backtest_predictions_long.empty:
        artifacts.backtest_predictions_long.to_csv(long_backtest_path, index=False)
    if artifacts.backtest_metrics_by_horizon is not None and not artifacts.backtest_metrics_by_horizon.empty:
        artifacts.backtest_metrics_by_horizon.to_csv(horizon_metrics_path, index=False)
    if artifacts.backtest_metrics_by_bucket is not None and not artifacts.backtest_metrics_by_bucket.empty:
        artifacts.backtest_metrics_by_bucket.to_csv(bucket_metrics_path, index=False)

    paths = {"history": history_path, "forecast": forecast_path, "submission": payload_path}
    if artifacts.backtest_summary is not None and not artifacts.backtest_summary.empty:
        paths["backtest"] = backtest_path
    if artifacts.selected_weights is not None:
        paths["weights"] = weights_path
    if artifacts.selected_bucket_weights is not None:
        paths["bucket_weights"] = bucket_weights_path
    if artifacts.backtest_predictions_long is not None and not artifacts.backtest_predictions_long.empty:
        paths["backtest_long"] = long_backtest_path
    if artifacts.backtest_metrics_by_horizon is not None and not artifacts.backtest_metrics_by_horizon.empty:
        paths["backtest_horizon"] = horizon_metrics_path
    if artifacts.backtest_metrics_by_bucket is not None and not artifacts.backtest_metrics_by_bucket.empty:
        paths["backtest_bucket"] = bucket_metrics_path
    return paths


def load_template_from_api(
    api_endpoint: str,
    access_token: str,
    scenario_name: str,
) -> list[dict]:
    url = urllib.parse.urljoin(api_endpoint, "competition/forecast_competition_template")
    url = f"{url}?{urllib.parse.urlencode({'scenario_name': scenario_name})}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Competition template response is not a list.")
    return data


def generate_forecast_from_api(
    access_token: str,
    scenario_name: str = DEFAULT_SCENARIO,
    api_server: str = DEFAULT_API_SERVER,
    horizon: int = DEFAULT_HORIZON,
) -> ForecastArtifacts:
    api_endpoint = build_api_endpoint(api_server)
    options_df = get_from_api(
        "data/dim/options",
        api_endpoint,
        access_token=access_token,
        scenario_name=scenario_name,
    )
    items_df = get_from_api(
        "data/dim/items",
        api_endpoint,
        access_token=access_token,
        scenario_name=scenario_name,
    )
    raw_sales = get_from_api(
        "data/dwd/option_sales_by_day",
        api_endpoint,
        access_token=access_token,
        scenario_name=scenario_name,
        limit=0,
    )

    history = normalize_history(raw_sales, options_df=options_df)
    raw_daily = build_raw_daily_panel(raw_sales)
    events_df = load_events_from_api(api_endpoint, access_token, scenario_name=scenario_name)
    backtest_summary, selected_weights, selected_bucket_weights, backtest_long, horizon_metrics, bucket_metrics = run_backtest(
        history=history,
        raw_daily=raw_daily,
        options_df=options_df,
        items_df=items_df,
        events_df=events_df,
        horizon=horizon,
    )
    forecast_panel = run_ensemble_forecast(
        history,
        raw_daily=raw_daily,
        options_df=options_df,
        items_df=items_df,
        events_df=events_df,
        horizon=horizon,
        wbaseline_weight=selected_weights["wbaseline"],
        etr_weight=selected_weights["etr"],
        lgbm_weight=selected_weights["lgbm"],
        topdown_weight=selected_weights["item_topdown"],
        bucket_weights=selected_bucket_weights,
    )
    template = load_template_from_api(api_endpoint, access_token, scenario_name=scenario_name)
    submission_payload = format_submission_payload(forecast_panel, template=template, horizon=horizon)
    return ForecastArtifacts(
        history=history,
        forecast_panel=forecast_panel,
        submission_payload=submission_payload,
        backtest_summary=backtest_summary,
        selected_weights=selected_weights,
        selected_bucket_weights=selected_bucket_weights,
        backtest_predictions_long=backtest_long,
        backtest_metrics_by_horizon=horizon_metrics,
        backtest_metrics_by_bucket=bucket_metrics,
    )


def generate_forecast_from_csv(history_csv: Path, horizon: int = DEFAULT_HORIZON) -> ForecastArtifacts:
    raw_history = pd.read_csv(history_csv)
    history = normalize_history(raw_history)
    raw_daily = build_raw_daily_panel(raw_history)
    synthetic_options = history[["option_id"]].drop_duplicates().assign(
        item_id=lambda df: df["option_id"],
        label_price=0.0,
    )
    synthetic_items = synthetic_options.assign(cate1_id="unknown", cate2_id="unknown")
    forecast_panel = run_ensemble_forecast(
        history,
        raw_daily=raw_daily,
        options_df=synthetic_options,
        items_df=synthetic_items,
        events_df=pd.DataFrame(columns=["date", "event_name", "event_type"]),
        horizon=horizon,
    )
    submission_payload = format_submission_payload(forecast_panel, horizon=horizon)
    return ForecastArtifacts(history=history, forecast_panel=forecast_panel, submission_payload=submission_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a baseline 28-day option-level forecast for the forecasting challenge."
    )
    parser.add_argument("--mode", choices=["api", "csv"], default="api")
    parser.add_argument("--access-token", default=os.getenv("RETAIL_ANALYTICS_ACCESS_TOKEN"))
    parser.add_argument("--api-server", default=DEFAULT_API_SERVER)
    parser.add_argument("--scenario-name", default=DEFAULT_SCENARIO)
    parser.add_argument("--history-csv", type=Path, help="Offline history CSV with option_id/date/quantity columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "api":
        if not args.access_token:
            raise ValueError("`--access-token` is required in api mode.")
        artifacts = generate_forecast_from_api(
            access_token=args.access_token,
            scenario_name=args.scenario_name,
            api_server=args.api_server,
            horizon=args.horizon,
        )
    else:
        if args.history_csv is None:
            raise ValueError("`--history-csv` is required in csv mode.")
        artifacts = generate_forecast_from_csv(args.history_csv, horizon=args.horizon)

    feature_date = str(pd.to_datetime(artifacts.history["date"]).max().date())
    paths = save_outputs(artifacts, output_dir=args.output_dir, feature_date=feature_date)

    print("Forecast generation completed.")
    print(f"Feature date: {feature_date}")
    print(f"History rows: {len(artifacts.history)}")
    print(f"Forecast rows: {len(artifacts.forecast_panel)}")
    if artifacts.selected_weights:
        print(f"Selected weights: {json.dumps(artifacts.selected_weights, ensure_ascii=False)}")
    if artifacts.selected_bucket_weights:
        print(f"Selected bucket weights: {json.dumps(artifacts.selected_bucket_weights, ensure_ascii=False)}")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
