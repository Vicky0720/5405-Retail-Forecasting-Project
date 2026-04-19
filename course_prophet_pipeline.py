from __future__ import annotations

import json
import os
import urllib.parse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from prophet import Prophet

from forecast_week1 import (
    DEFAULT_API_SERVER,
    DEFAULT_HORIZON,
    DEFAULT_SCENARIO,
    build_api_endpoint,
    format_submission_payload,
    get_from_api,
    load_events_from_api,
    load_template_from_api,
    normalize_history,
    weighted_1mape,
)


OUTPUT_DIR = Path("outputs")
ORIGINS = [
    pd.Timestamp("2022-07-14"),
    pd.Timestamp("2022-07-21"),
    pd.Timestamp("2022-07-28"),
    pd.Timestamp("2022-08-04"),
    pd.Timestamp("2022-08-11"),
    pd.Timestamp("2022-08-18"),
]
HOLDOUT_ORIGINS = 2
RESIDUAL_BETAS = [0.3, 0.5, 0.7, 0.9]
SHAPE_STRENGTHS = [0.30, 0.45, 0.60]
SHAPE_SCORE_TOLERANCE = 0.035
LEVEL_RECONCILE_START = 15
LEVEL_RECONCILE_PROFILES = {
    "off": {"1-7": 0.0, "8-14": 0.0, "15-21": 0.0, "22-28": 0.0},
    "light": {"1-7": 0.0, "8-14": 0.0, "15-21": 0.18, "22-28": 0.32},
    "medium": {"1-7": 0.0, "8-14": 0.0, "15-21": 0.32, "22-28": 0.55},
    "strong": {"1-7": 0.0, "8-14": 0.08, "15-21": 0.46, "22-28": 0.72},
}
HYBRID_RESIDUAL_BETAS = [0.5, 0.7, 0.9]
HYBRID_SHAPE_STRENGTH = 0.30
HYBRID_LEVEL_PROFILE = "light"
HYBRID_SCORE_TOLERANCE = 0.045


def make_holidays(events: pd.DataFrame) -> pd.DataFrame | None:
    holidays = []
    if events.empty:
        return None
    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "event_type" not in frame.columns:
        frame["event_type"] = "event"
    for event_type, group in frame.groupby("event_type"):
        name = str(event_type) if str(event_type) else "event"
        holidays.append(
            pd.DataFrame(
                {
                    "holiday": name,
                    "ds": pd.to_datetime(group["date"].unique()),
                    "lower_window": -3,
                    "upper_window": 2,
                }
            )
        )
    return pd.concat(holidays, ignore_index=True) if holidays else None


def prophet_forecast(series: pd.Series, origin: pd.Timestamp, horizon: int, holidays: pd.DataFrame | None) -> np.ndarray:
    train = series[series.index <= origin].astype(float).reset_index()
    train.columns = ["ds", "y"]
    train["y"] = train["y"].clip(lower=0.0)
    if train["y"].sum() <= 0 or train["y"].nunique() <= 2:
        return seasonal_mix_forecast(series, origin, pd.date_range(origin + pd.Timedelta(days=1), periods=horizon))

    try:
        model = Prophet(
            n_changepoints=0,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            holidays=holidays,
            holidays_mode="multiplicative",
        )
        model.fit(train)
        future = model.make_future_dataframe(periods=horizon, include_history=False)
        pred = model.predict(future)["yhat"].to_numpy(dtype=float)
        return np.clip(pred, 0.0, None)
    except Exception:
        return seasonal_mix_forecast(series, origin, pd.date_range(origin + pd.Timedelta(days=1), periods=horizon))


def seasonal_mix_forecast(series: pd.Series, origin: pd.Timestamp, future_dates: pd.DatetimeIndex) -> np.ndarray:
    known = {pd.Timestamp(k): float(v) for k, v in series.sort_index().items() if pd.Timestamp(k) <= origin}
    fallback = float(pd.Series(list(known.values())).tail(28).mean()) if known else 0.0
    values = []
    for target_date in pd.to_datetime(future_dates):
        candidates = []
        weights = []
        for lag, weight in [(7, 0.5), (14, 0.3), (28, 0.2)]:
            lag_date = target_date - pd.Timedelta(days=lag)
            if lag_date in known:
                candidates.append(known[lag_date])
                weights.append(weight)
        pred = float(np.average(candidates, weights=weights)) if candidates else fallback
        pred = max(pred, 0.0)
        known[target_date] = pred
        values.append(pred)
    return np.array(values)


def recent_share(df: pd.DataFrame, origin: pd.Timestamp, group_cols: list[str], child_col: str, window: int = 28) -> pd.DataFrame:
    cut = df[(df["date"] <= origin) & (df["date"] > origin - pd.Timedelta(days=window))]
    shares = cut.groupby(group_cols + [child_col], as_index=False)["target"].sum()
    denom = shares.groupby(group_cols)["target"].transform("sum")
    shares["ratio"] = np.where(denom > 0, shares["target"] / denom, 0.0)
    return shares[group_cols + [child_col, "ratio"]]


def topdown_forecast(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    parent_col: str,
    child_col: str,
    parent_method: str,
    holidays: pd.DataFrame | None,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
    parent_daily = df.groupby([parent_col, "date"], as_index=False)["target"].sum()
    parts = []
    for parent_id, group in parent_daily.groupby(parent_col):
        series = group.set_index("date")["target"].sort_index()
        if parent_method == "prophet":
            pred = prophet_forecast(series, origin, horizon, holidays)
        elif parent_method == "mix71428":
            pred = seasonal_mix_forecast(series, origin, future)
        else:
            raise ValueError(parent_method)
        parts.append(pd.DataFrame({parent_col: parent_id, "target_date": future, "parent_forecast": pred}))
    parent_forecast = pd.concat(parts, ignore_index=True)

    shares = recent_share(df, origin, [parent_col], child_col, window=28)
    output = parent_forecast.merge(shares, on=parent_col, how="left")
    counts = df[[parent_col, child_col]].drop_duplicates().groupby(parent_col).size().rename("n").reset_index()
    output = output.merge(counts, on=parent_col, how="left")
    output["ratio"] = output["ratio"].fillna(1.0 / output["n"])
    output["forecast"] = output["parent_forecast"] * output["ratio"]
    return output[[child_col, "target_date", "forecast"]].rename(columns={child_col: "option_id"})


def bottomup_forecast(df: pd.DataFrame, origin: pd.Timestamp, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
    parts = []
    for option_id, group in df.groupby("option_id"):
        pred = seasonal_mix_forecast(group.set_index("date")["target"].sort_index(), origin, future)
        parts.append(pd.DataFrame({"option_id": option_id, "target_date": future, "forecast": pred}))
    return pd.concat(parts, ignore_index=True)


def actual_for(df: pd.DataFrame, origin: pd.Timestamp, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    future = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
    return df[df["date"].isin(future)][["option_id", "date", "target"]].rename(
        columns={"date": "target_date", "target": "actual"}
    )


def build_candidates(df: pd.DataFrame, origin: pd.Timestamp, holidays: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    return {
        "cate2_prophet": topdown_forecast(df, origin, "cate2_id", "option_id", "prophet", holidays),
        "item_prophet": topdown_forecast(df, origin, "item_id", "option_id", "prophet", holidays),
        "cate2_mix": topdown_forecast(df, origin, "cate2_id", "option_id", "mix71428", holidays),
        "bottomup_mix": bottomup_forecast(df, origin),
    }


def merge_candidates(actual: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = actual.copy()
    for name, frame in candidates.items():
        merged = merged.merge(
            frame.rename(columns={"forecast": f"forecast_{name}"}),
            on=["option_id", "target_date"],
            how="left",
        )
        merged[f"forecast_{name}"] = merged[f"forecast_{name}"].fillna(0.0)
    return merged


def search_weights(selection: pd.DataFrame, model_names: list[str], step: float = 0.1) -> dict[str, float]:
    best_score = -np.inf
    best_weights = {name: 0.0 for name in model_names}
    units = int(round(1.0 / step))
    for weights_int in product(range(units + 1), repeat=len(model_names)):
        if sum(weights_int) != units:
            continue
        weights = {name: value * step for name, value in zip(model_names, weights_int)}
        pred = sum(weights[name] * selection[f"forecast_{name}"].to_numpy(dtype=float) for name in model_names)
        score = weighted_1mape(selection["actual"], pred)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights


def horizon_bucket(horizon: int) -> str:
    if horizon <= 7:
        return "1-7"
    if horizon <= 14:
        return "8-14"
    if horizon <= 21:
        return "15-21"
    return "22-28"


def split_origins(origins: list[pd.Timestamp]) -> tuple[set[str], set[str]]:
    holdout = origins[-HOLDOUT_ORIGINS:] if HOLDOUT_ORIGINS > 0 else []
    selection = origins[: len(origins) - len(holdout)]
    return {str(x.date()) for x in selection}, {str(x.date()) for x in holdout}


def rolling_origins_for_feature(feature_date: pd.Timestamp, count: int = 6) -> list[pd.Timestamp]:
    latest_origin = feature_date - pd.Timedelta(days=DEFAULT_HORIZON)
    return [latest_origin - pd.Timedelta(days=7 * i) for i in range(count - 1, -1, -1)]


def add_ensemble_forecast(frame: pd.DataFrame, weights: dict[str, float], model_names: list[str]) -> pd.DataFrame:
    output = frame.copy()
    output["forecast_ensemble"] = sum(
        weights[name] * output[f"forecast_{name}"].to_numpy(dtype=float) for name in model_names
    )
    return output


def residual_correct_panel(
    forecast_frame: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    beta: float,
) -> pd.DataFrame:
    output = forecast_frame.copy()
    actual_total = history[history["date"] <= origin].groupby("date")["target"].sum().sort_index()
    base_total = output.groupby("target_date")["forecast"].sum().sort_index()
    smooth = 0.6 * actual_total.rolling(7, min_periods=1).mean() + 0.4 * actual_total.rolling(14, min_periods=1).mean()
    residual = actual_total - smooth

    adjustments = []
    for i, target_date in enumerate(base_total.index, start=1):
        candidates = []
        weights = []
        for lag, weight in [(7, 0.45), (14, 0.30), (21, 0.15), (28, 0.10)]:
            lag_date = target_date - pd.Timedelta(days=lag)
            if lag_date in residual.index:
                candidates.append(float(residual.loc[lag_date]))
                weights.append(weight)
        residual_adjustment = float(np.average(candidates, weights=weights)) if candidates else 0.0
        decay = 1.0 - 0.25 * (i - 1) / max(len(base_total) - 1, 1)
        adjustments.append(beta * decay * residual_adjustment)

    lower = max(0.0, float(actual_total.tail(56).quantile(0.10)))
    upper = float(actual_total.tail(56).quantile(0.95) + 15.0)
    target_total = (base_total + np.array(adjustments)).clip(lower=lower, upper=upper)
    scale = (target_total / base_total.replace(0, np.nan)).fillna(1.0)
    output["forecast"] = [row.forecast * scale.loc[row.target_date] for row in output.itertuples()]
    return output


def reconcile_far_horizon_level(
    target_total: pd.Series,
    total_history: pd.Series,
    origin: pd.Timestamp,
    blend_profile: dict[str, float],
) -> pd.Series:
    output = target_total.copy()
    recent_14 = float(total_history.tail(14).mean())
    recent_28 = float(total_history.tail(28).mean())
    recent_anchor = 0.65 * recent_14 + 0.35 * recent_28
    lag_weights = [(7, 0.42), (14, 0.25), (21, 0.16), (28, 0.10), (35, 0.05), (42, 0.02)]

    for target_date, value in target_total.items():
        horizon = max((target_date - origin).days, 1)
        if horizon < LEVEL_RECONCILE_START:
            continue
        max_blend = float(blend_profile.get(horizon_bucket(horizon), 0.0))
        if max_blend <= 0:
            continue
        same_weekday_values = []
        weights = []
        for lag, weight in lag_weights:
            lag_date = target_date - pd.Timedelta(days=lag)
            if lag_date in total_history.index:
                same_weekday_values.append(float(total_history.loc[lag_date]))
                weights.append(weight)
        weekday_anchor = float(np.average(same_weekday_values, weights=weights)) if same_weekday_values else recent_anchor
        anchor = 0.75 * weekday_anchor + 0.25 * recent_anchor
        ramp = np.clip(
            (horizon - LEVEL_RECONCILE_START + 1) / (DEFAULT_HORIZON - LEVEL_RECONCILE_START + 1),
            0.0,
            1.0,
        )
        blend = max_blend * ramp
        if value < anchor:
            output.loc[target_date] = value + blend * (anchor - value)
    return output


def reconcile_group_levels(
    forecast_frame: pd.DataFrame,
    history_cut: pd.DataFrame,
    origin: pd.Timestamp,
    group_col: str,
    blend_profile: dict[str, float],
) -> pd.DataFrame:
    if group_col not in history_cut.columns:
        return forecast_frame

    option_groups = history_cut[["option_id", group_col]].dropna().drop_duplicates("option_id")
    if option_groups.empty:
        return forecast_frame

    enriched = forecast_frame.merge(option_groups, on="option_id", how="left")
    history_grouped = history_cut.dropna(subset=[group_col]).groupby([group_col, "date"])["target"].sum()
    for group_id, group in enriched.dropna(subset=[group_col]).groupby(group_col):
        if group_id not in history_grouped.index.get_level_values(0):
            continue
        base_total = group.groupby("target_date")["forecast"].sum().sort_index()
        group_history = history_grouped.loc[group_id].sort_index()
        if len(group_history) < 28 or group_history.tail(28).sum() <= 0:
            continue
        target_total = reconcile_far_horizon_level(base_total, group_history, origin, blend_profile)
        recent_group = group_history.tail(56)
        upper = float(recent_group.quantile(0.98) + max(2.0, recent_group.mean() * 0.10))
        target_total = target_total.clip(lower=0.0, upper=upper)
        scale = (target_total / base_total.replace(0, np.nan)).fillna(1.0)
        idx = enriched[group_col] == group_id
        enriched.loc[idx, "forecast"] = [
            row.forecast * scale.loc[row.target_date] for row in enriched.loc[idx].itertuples()
        ]
    return enriched[["option_id", "target_date", "forecast"]]


def shape_correct_panel(
    forecast_frame: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    strength: float,
    level_profile: str = "medium",
) -> pd.DataFrame:
    """Inject same-weekday residual shape while keeping the base model's level anchored."""
    output = forecast_frame.copy()
    output["target_date"] = pd.to_datetime(output["target_date"])
    history_cut = history[history["date"] <= origin].copy()
    history_cut["date"] = pd.to_datetime(history_cut["date"])
    blend_profile = LEVEL_RECONCILE_PROFILES[level_profile]

    option_history = history_cut[["option_id", "date", "target"]].copy()
    option_history["smooth"] = (
        option_history.groupby("option_id")["target"].transform(lambda s: 0.45 * s.rolling(7, min_periods=1).mean() + 0.55 * s.rolling(28, min_periods=1).mean())
    )
    option_history["ratio"] = (option_history["target"] / option_history["smooth"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.35, 2.25)
    option_ratio = option_history.set_index(["option_id", "date"])["ratio"]

    total_history = history_cut.groupby("date")["target"].sum().sort_index()
    total_smooth = 0.45 * total_history.rolling(7, min_periods=1).mean() + 0.55 * total_history.rolling(28, min_periods=1).mean()
    total_ratio = (total_history / total_smooth.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.55, 1.75)

    lag_weights = [(7, 0.45), (14, 0.30), (21, 0.15), (28, 0.10)]
    option_factors = []
    for row in output.itertuples():
        ratios = []
        weights = []
        for lag, weight in lag_weights:
            key = (row.option_id, row.target_date - pd.Timedelta(days=lag))
            if key in option_ratio.index:
                ratios.append(float(option_ratio.loc[key]))
                weights.append(weight)
        ratio = float(np.average(ratios, weights=weights)) if ratios else 1.0
        horizon = max((row.target_date - origin).days, 1)
        decay = 1.0 - 0.30 * (horizon - 1) / (DEFAULT_HORIZON - 1)
        option_factors.append(float(np.clip(1.0 + strength * decay * (ratio - 1.0), 0.65, 1.45)))
    output["forecast"] = output["forecast"] * np.array(option_factors)
    output = reconcile_group_levels(output, history_cut, origin, "cate2_id", blend_profile)
    output = reconcile_group_levels(output, history_cut, origin, "item_id", blend_profile)

    base_total = output.groupby("target_date")["forecast"].sum().sort_index()
    target_totals = []
    for target_date in base_total.index:
        ratios = []
        weights = []
        for lag, weight in lag_weights:
            lag_date = target_date - pd.Timedelta(days=lag)
            if lag_date in total_ratio.index:
                ratios.append(float(total_ratio.loc[lag_date]))
                weights.append(weight)
        ratio = float(np.average(ratios, weights=weights)) if ratios else 1.0
        horizon = max((target_date - origin).days, 1)
        decay = 1.0 - 0.30 * (horizon - 1) / (DEFAULT_HORIZON - 1)
        total_factor = float(np.clip(1.0 + strength * decay * (ratio - 1.0), 0.78, 1.28))
        target_totals.append(base_total.loc[target_date] * total_factor)

    target_total = pd.Series(target_totals, index=base_total.index)
    recent_total = total_history.tail(56)
    target_total = reconcile_far_horizon_level(target_total, total_history, origin, blend_profile)
    soft_lower = max(0.0, float(recent_total.quantile(0.05)))
    upper = float(recent_total.quantile(0.97) + 12.0)
    below_floor = target_total < soft_lower
    target_total.loc[below_floor] = soft_lower - 0.45 * (soft_lower - target_total.loc[below_floor])
    target_total = target_total.clip(lower=0.0, upper=upper)
    scale = (target_total / base_total.replace(0, np.nan)).fillna(1.0)
    output["forecast"] = [row.forecast * scale.loc[row.target_date] for row in output.itertuples()]
    return output[["option_id", "target_date", "forecast"]]


def horizon_hybrid_panel(
    residual_frame: pd.DataFrame,
    item_frame: pd.DataFrame,
    shape_frame: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    merged = residual_frame.rename(columns={"forecast": "forecast_residual"}).merge(
        item_frame.rename(columns={"forecast": "forecast_item"}),
        on=["option_id", "target_date"],
        how="inner",
    ).merge(
        shape_frame.rename(columns={"forecast": "forecast_shape"}),
        on=["option_id", "target_date"],
        how="inner",
    )
    horizon = (pd.to_datetime(merged["target_date"]) - origin).dt.days
    merged["forecast"] = np.select(
        [
            horizon <= 7,
            horizon <= 14,
        ],
        [
            merged["forecast_residual"],
            0.65 * merged["forecast_residual"] + 0.35 * merged["forecast_item"],
        ],
        default=merged["forecast_shape"],
    )
    return merged[["option_id", "target_date", "forecast"]]


def add_horizon_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["horizon"] = (
        pd.to_datetime(output["target_date"]) - pd.to_datetime(output["origin"])
    ).dt.days.astype(int)
    output["horizon_bucket"] = output["horizon"].map(horizon_bucket)
    return output


def fit_bias_corrections(selection: pd.DataFrame, mapping: pd.DataFrame) -> dict[str, object]:
    enriched = add_horizon_columns(selection).merge(mapping, on="option_id", how="left")
    global_factor = float(enriched["actual"].sum() / enriched["forecast"].sum()) if enriched["forecast"].sum() > 0 else 1.0
    global_factor = float(np.clip(global_factor, 0.75, 1.35))

    bucket = (
        enriched.groupby("horizon_bucket")
        .apply(lambda g: g["actual"].sum() / g["forecast"].sum() if g["forecast"].sum() > 0 else 1.0, include_groups=False)
        .clip(0.75, 1.35)
        .to_dict()
    )
    item = (
        enriched.groupby("item_id")
        .apply(lambda g: g["actual"].sum() / g["forecast"].sum() if g["forecast"].sum() > 0 else 1.0, include_groups=False)
        .clip(0.75, 1.35)
        .to_dict()
    )
    return {"global": global_factor, "horizon_bucket": bucket, "item": item}


def apply_bias_correction(frame: pd.DataFrame, mapping: pd.DataFrame, corrections: dict[str, object], mode: str) -> pd.DataFrame:
    output = frame.copy()
    if mode == "global":
        output["forecast"] = output["forecast"] * float(corrections["global"])
        return output[["option_id", "target_date", "forecast"]]
    enriched = add_horizon_columns(output).merge(mapping, on="option_id", how="left")
    if mode == "horizon":
        factors = corrections["horizon_bucket"]
        enriched["factor"] = enriched["horizon_bucket"].map(factors).fillna(1.0)
    elif mode == "item":
        factors = corrections["item"]
        enriched["factor"] = enriched["item_id"].map(factors).fillna(1.0)
    else:
        raise ValueError(mode)
    enriched["forecast"] = enriched["forecast"] * enriched["factor"]
    return enriched[["option_id", "target_date", "forecast"]]


def one_minus_ape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    actual_arr = pd.to_numeric(actual, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    forecast_arr = pd.to_numeric(forecast, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score = np.zeros_like(actual_arr, dtype=float)
    positive = actual_arr > 0
    score[positive] = 1.0 - np.abs(actual_arr[positive] - forecast_arr[positive]) / actual_arr[positive]
    return pd.Series(score, index=actual.index)


def summarize_accuracy(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def _metrics(group: pd.DataFrame) -> pd.Series:
        actual = group["actual"].astype(float)
        forecast = group["forecast"].astype(float)
        abs_error = (actual - forecast).abs()
        signed_error = forecast - actual
        weight = actual.clip(lower=0.0)
        weighted_error = abs_error.sum()
        actual_sum = weight.sum()
        weighted_score = weighted_1mape(actual, forecast)
        return pd.Series(
            {
                "score": weighted_score,
                "actual_sum": float(actual_sum),
                "forecast_sum": float(forecast.sum()),
                "abs_error_sum": float(weighted_error),
                "signed_error_sum": float(signed_error.sum()),
                "bias_ratio": float(signed_error.sum() / actual_sum) if actual_sum > 0 else 0.0,
                "n_rows": len(group),
            }
        )

    if not group_cols:
        return pd.DataFrame([_metrics(frame)])
    return (
        frame.groupby(group_cols, as_index=False)
        .apply(lambda g: _metrics(g), include_groups=False)
        .reset_index(drop=True)
    )


def add_accuracy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["horizon"] = (
        pd.to_datetime(output["target_date"]) - pd.to_datetime(output["origin"])
    ).dt.days.astype(int)
    output["abs_error"] = (output["actual"] - output["forecast"]).abs()
    output["signed_error"] = output["forecast"] - output["actual"]
    output["one_minus_ape"] = one_minus_ape(output["actual"], output["forecast"])
    return output


def save_accuracy_attribution(
    long_panel: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    feature_tag: str,
) -> None:
    enriched = long_panel.merge(mapping, on="option_id", how="left")
    enriched = add_accuracy_columns(enriched)
    enriched.to_csv(output_dir / f"course_prophet_backtest_predictions_long_{feature_tag}.csv", index=False)

    leaderboard = summarize_accuracy(enriched, ["model"]).sort_values("score", ascending=False)
    leaderboard.to_csv(output_dir / f"course_prophet_model_leaderboard_{feature_tag}.csv", index=False)
    split_leaderboard = summarize_accuracy(enriched, ["split", "model"]).sort_values(["split", "score"], ascending=[True, False])
    split_leaderboard.to_csv(output_dir / f"course_prophet_model_leaderboard_by_split_{feature_tag}.csv", index=False)

    for name, group_cols in {
        "day": ["model", "target_date"],
        "horizon": ["model", "horizon"],
        "option": ["model", "option_id"],
        "item": ["model", "item_id"],
        "cate2": ["model", "cate2_id"],
    }.items():
        summary = summarize_accuracy(enriched, group_cols)
        summary.to_csv(output_dir / f"course_prophet_accuracy_by_{name}_{feature_tag}.csv", index=False)

    ensemble = enriched[enriched["model"] == "ensemble"].copy()
    contribution_rows = []
    total_abs_error = ensemble["abs_error"].sum()
    for name, group_col in [("day", "target_date"), ("option", "option_id"), ("item", "item_id"), ("cate2", "cate2_id")]:
        grouped = ensemble.groupby(group_col, as_index=False).agg(
            actual_sum=("actual", "sum"),
            forecast_sum=("forecast", "sum"),
            abs_error_sum=("abs_error", "sum"),
            signed_error_sum=("signed_error", "sum"),
        )
        grouped["contribution_share"] = (
            grouped["abs_error_sum"] / total_abs_error if total_abs_error > 0 else 0.0
        )
        grouped["attribution_level"] = name
        grouped = grouped.rename(columns={group_col: "key"})
        contribution_rows.append(grouped)
    pd.concat(contribution_rows, ignore_index=True).to_csv(
        output_dir / f"course_prophet_error_contribution_{feature_tag}.csv",
        index=False,
    )


def run() -> None:
    token = os.getenv("RETAIL_ANALYTICS_ACCESS_TOKEN")
    if not token:
        raise ValueError("Set RETAIL_ANALYTICS_ACCESS_TOKEN before running this script.")

    api_endpoint = build_api_endpoint(DEFAULT_API_SERVER)
    options = get_from_api("data/dim/options", api_endpoint, access_token=token, scenario_name=DEFAULT_SCENARIO)
    items = get_from_api("data/dim/items", api_endpoint, access_token=token, scenario_name=DEFAULT_SCENARIO)
    raw = get_from_api(
        "data/dwd/option_sales_by_day",
        api_endpoint,
        access_token=token,
        scenario_name=DEFAULT_SCENARIO,
        limit=0,
    )
    events = load_events_from_api(api_endpoint, token, scenario_name=DEFAULT_SCENARIO)
    holidays = make_holidays(events)

    history = normalize_history(raw, options_df=options)
    mapping = options[["option_id", "item_id"]].merge(items[["item_id", "cate2_id", "cate1_id"]], on="item_id", how="left")
    panel = history.merge(mapping, on="option_id", how="left")
    feature_date = pd.to_datetime(panel["date"]).max()
    feature_tag = str(feature_date.date()).replace("-", "")
    model_names = ["cate2_prophet", "item_prophet", "cate2_mix", "bottomup_mix"]
    origins = rolling_origins_for_feature(feature_date)
    selection_origin_set, holdout_origin_set = split_origins(origins)

    backtest_panels = []
    for origin in origins:
        train_panel = panel[panel["date"] <= origin]
        actual = actual_for(panel, origin)
        merged = merge_candidates(actual, build_candidates(train_panel, origin, holidays))
        merged["origin"] = str(origin.date())
        merged["split"] = "holdout" if str(origin.date()) in holdout_origin_set else "selection"
        backtest_panels.append(merged)

    backtest_wide = pd.concat(backtest_panels, ignore_index=True)
    selection = backtest_wide[backtest_wide["split"] == "selection"].copy()
    weights = search_weights(selection, model_names)
    backtest_wide = add_ensemble_forecast(backtest_wide, weights, model_names)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"course_prophet_pipeline_weights_{feature_tag}.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )

    long_parts = []
    for name in model_names:
        part = backtest_wide[["origin", "split", "option_id", "target_date", "actual", f"forecast_{name}"]].rename(
            columns={f"forecast_{name}": "forecast"}
        )
        part["model"] = name
        long_parts.append(part)
    ensemble_part = backtest_wide[["origin", "split", "option_id", "target_date", "actual"]].copy()
    ensemble_part["forecast"] = backtest_wide["forecast_ensemble"]
    ensemble_part["model"] = "ensemble"
    long_parts.append(ensemble_part)

    for base_model in ["item_prophet", "ensemble"]:
        base_source = long_parts[model_names.index(base_model)] if base_model in model_names else ensemble_part
        base_source = base_source[["origin", "split", "option_id", "target_date", "actual", "forecast"]].copy()
        for strength in SHAPE_STRENGTHS:
            for level_profile in LEVEL_RECONCILE_PROFILES:
                corrected_parts = []
                for origin_text, frame in base_source.groupby("origin"):
                    origin = pd.Timestamp(origin_text)
                    tmp = shape_correct_panel(
                        frame[["option_id", "target_date", "forecast"]],
                        history=panel,
                        origin=origin,
                        strength=strength,
                        level_profile=level_profile,
                    )
                    tmp = frame[["origin", "split", "option_id", "target_date", "actual"]].merge(
                        tmp,
                        on=["option_id", "target_date"],
                        how="left",
                    )
                    tmp["model"] = f"{base_model}_shape_{strength:.2f}_level_{level_profile}"
                    corrected_parts.append(tmp)
                long_parts.append(pd.concat(corrected_parts, ignore_index=True))

    residual_source = ensemble_part[["origin", "split", "option_id", "target_date", "actual", "forecast"]].copy()
    hybrid_parts_by_beta = {beta: [] for beta in HYBRID_RESIDUAL_BETAS}
    for beta in RESIDUAL_BETAS:
        corrected_parts = []
        for origin_text, frame in residual_source.groupby("origin"):
            origin = pd.Timestamp(origin_text)
            tmp = residual_correct_panel(
                frame[["option_id", "target_date", "forecast"]],
                history=history,
                origin=origin,
                beta=beta,
            )
            tmp = frame[["origin", "split", "option_id", "target_date", "actual"]].merge(
                tmp,
                on=["option_id", "target_date"],
                how="left",
            )
            tmp["model"] = f"ensemble_residual_beta_{beta:.1f}"
            corrected_parts.append(tmp)
            if beta in hybrid_parts_by_beta:
                item_frame = backtest_wide[backtest_wide["origin"] == origin_text][
                    ["option_id", "target_date", "forecast_item_prophet"]
                ].rename(columns={"forecast_item_prophet": "forecast"})
                shape_frame = shape_correct_panel(
                    item_frame,
                    history=panel,
                    origin=origin,
                    strength=HYBRID_SHAPE_STRENGTH,
                    level_profile=HYBRID_LEVEL_PROFILE,
                )
                hybrid = horizon_hybrid_panel(
                    residual_frame=tmp[["option_id", "target_date", "forecast"]],
                    item_frame=item_frame,
                    shape_frame=shape_frame,
                    origin=origin,
                )
                hybrid = frame[["origin", "split", "option_id", "target_date", "actual"]].merge(
                    hybrid,
                    on=["option_id", "target_date"],
                    how="left",
                )
                hybrid["model"] = f"hybrid_short_residual_beta_{beta:.1f}"
                hybrid_parts_by_beta[beta].append(hybrid)
        long_parts.append(pd.concat(corrected_parts, ignore_index=True))
    for beta, parts in hybrid_parts_by_beta.items():
        if parts:
            long_parts.append(pd.concat(parts, ignore_index=True))

    selection_ensemble = ensemble_part[ensemble_part["split"] == "selection"].copy()
    corrections = fit_bias_corrections(selection_ensemble, mapping)
    (OUTPUT_DIR / f"course_prophet_bias_corrections_{feature_tag}.json").write_text(
        json.dumps(corrections, indent=2, default=str),
        encoding="utf-8",
    )
    for mode in ["global", "horizon", "item"]:
        corrected_parts = []
        for _, frame in ensemble_part.groupby("origin"):
            tmp = apply_bias_correction(
                frame[["origin", "option_id", "target_date", "forecast"]],
                mapping=mapping,
                corrections=corrections,
                mode=mode,
            )
            tmp = frame[["origin", "split", "option_id", "target_date", "actual"]].merge(
                tmp,
                on=["option_id", "target_date"],
                how="left",
            )
            tmp["model"] = f"ensemble_bias_{mode}"
            corrected_parts.append(tmp)
        long_parts.append(pd.concat(corrected_parts, ignore_index=True))

    full_long_panel = pd.concat(long_parts, ignore_index=True)
    backtest_summary = summarize_accuracy(full_long_panel, ["origin", "split", "model"])
    split_summary = summarize_accuracy(full_long_panel, ["split", "model"])
    split_summary["origin"] = split_summary["split"].map({"selection": "ALL_SELECTION", "holdout": "ALL_HOLDOUT"})
    backtest_summary = pd.concat(
        [
            backtest_summary,
            split_summary[["origin", "split", "model", "score", "actual_sum", "forecast_sum", "abs_error_sum", "signed_error_sum", "bias_ratio", "n_rows"]],
        ],
        ignore_index=True,
    )
    backtest_summary.to_csv(OUTPUT_DIR / f"course_prophet_pipeline_backtest_{feature_tag}.csv", index=False)
    save_accuracy_attribution(
        long_panel=full_long_panel,
        mapping=mapping,
        output_dir=OUTPUT_DIR,
        feature_tag=feature_tag,
    )

    holdout_scores = split_summary[split_summary["split"] == "holdout"].sort_values("score", ascending=False)
    if holdout_scores.empty:
        best_score_model = str(split_summary[split_summary["split"] == "selection"].sort_values("score", ascending=False).iloc[0]["model"])
        best_model = best_score_model
    else:
        best_score_model = str(holdout_scores.iloc[0]["model"])
        best_score = float(holdout_scores.iloc[0]["score"])
        hybrid_scores = holdout_scores[holdout_scores["model"].str.startswith("hybrid_short_residual_beta_")].copy()
        hybrid_scores = hybrid_scores[hybrid_scores["score"] >= best_score - HYBRID_SCORE_TOLERANCE]
        shape_scores = holdout_scores[holdout_scores["model"].str.contains("_shape_", regex=False)].copy()
        shape_scores = shape_scores[shape_scores["score"] >= best_score - SHAPE_SCORE_TOLERANCE]
        reconciled_shape_scores = shape_scores[~shape_scores["model"].str.endswith("_level_off")]
        if not hybrid_scores.empty:
            best_model = str(hybrid_scores.iloc[0]["model"])
        elif not reconciled_shape_scores.empty:
            best_model = str(reconciled_shape_scores.iloc[0]["model"])
        else:
            best_model = str(shape_scores.iloc[0]["model"]) if not shape_scores.empty else best_score_model
    (OUTPUT_DIR / f"course_prophet_selected_model_{feature_tag}.json").write_text(
        json.dumps(
            {
                "selected_score_model": best_score_model,
                "selected_submission_model": best_model,
                "selection_policy": "Prefer short-horizon residual hybrids within tolerance; otherwise prefer the best non-off hierarchical level reconciliation within tolerance.",
                "hybrid_score_tolerance": HYBRID_SCORE_TOLERANCE,
                "shape_score_tolerance": SHAPE_SCORE_TOLERANCE,
                "selection_origins": sorted(selection_origin_set),
                "holdout_origins": sorted(holdout_origin_set),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    candidates = build_candidates(panel, feature_date, holidays)
    final = None
    for name, frame in candidates.items():
        tmp = frame.rename(columns={"forecast": f"forecast_{name}"})
        final = tmp if final is None else final.merge(tmp, on=["option_id", "target_date"], how="inner")
    final["forecast"] = sum(weights[name] * final[f"forecast_{name}"] for name in model_names)
    final_panel = final[["option_id", "target_date", "forecast"]].sort_values(["option_id", "target_date"])
    if best_model.startswith("hybrid_short_residual_beta_"):
        beta = float(best_model.rsplit("_", 1)[-1])
        residual_panel = residual_correct_panel(final_panel, history=history, origin=feature_date, beta=beta)
        item_panel = final[["option_id", "target_date", "forecast_item_prophet"]].rename(
            columns={"forecast_item_prophet": "forecast"}
        )
        shape_panel = shape_correct_panel(
            item_panel,
            history=panel,
            origin=feature_date,
            strength=HYBRID_SHAPE_STRENGTH,
            level_profile=HYBRID_LEVEL_PROFILE,
        )
        final_panel = horizon_hybrid_panel(
            residual_frame=residual_panel[["option_id", "target_date", "forecast"]],
            item_frame=item_panel,
            shape_frame=shape_panel,
            origin=feature_date,
        )
    elif "_shape_" in best_model:
        shape_name, level_profile = best_model.rsplit("_level_", 1)
        base_model, strength_text = shape_name.rsplit("_shape_", 1)
        strength = float(strength_text)
        if base_model == "ensemble":
            base_panel = final_panel
        else:
            source_col = f"forecast_{base_model}"
            base_panel = final[["option_id", "target_date", source_col]].rename(columns={source_col: "forecast"})
        final_panel = shape_correct_panel(
            base_panel,
            history=panel,
            origin=feature_date,
            strength=strength,
            level_profile=level_profile,
        )
    elif best_model.startswith("ensemble_residual_beta_"):
        beta = float(best_model.rsplit("_", 1)[-1])
        final_panel = residual_correct_panel(final_panel, history=history, origin=feature_date, beta=beta)
    elif best_model == "ensemble_bias_global":
        final_panel = apply_bias_correction(final_panel, mapping=mapping, corrections=corrections, mode="global")
    elif best_model == "ensemble_bias_horizon":
        final_tmp = final_panel.copy()
        final_tmp["origin"] = str(feature_date.date())
        final_panel = apply_bias_correction(final_tmp, mapping=mapping, corrections=corrections, mode="horizon")
    elif best_model == "ensemble_bias_item":
        final_tmp = final_panel.copy()
        final_tmp["origin"] = str(feature_date.date())
        final_panel = apply_bias_correction(final_tmp, mapping=mapping, corrections=corrections, mode="item")
    elif best_model != "ensemble":
        source_col = f"forecast_{best_model}"
        if source_col in final.columns:
            final_panel = final[["option_id", "target_date", source_col]].rename(columns={source_col: "forecast"})

    final_panel = final_panel.sort_values(["option_id", "target_date"])
    final_panel.assign(target_date=final_panel["target_date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUTPUT_DIR / f"forecast_option_day_{feature_tag}_prophet_pipeline_selected.csv",
        index=False,
    )

    template = load_template_from_api(api_endpoint, token, scenario_name=DEFAULT_SCENARIO)
    payload = format_submission_payload(final_panel, template=template, horizon=DEFAULT_HORIZON)
    (OUTPUT_DIR / f"forecast_submission_{feature_tag}_prophet_pipeline_selected.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    url = urllib.parse.urljoin(api_endpoint, "competition/forecast_competition_input")
    url = f"{url}?{urllib.parse.urlencode({'scenario_name': DEFAULT_SCENARIO})}"
    response = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=120)
    response.raise_for_status()
    verify = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120).json()
    print("Feature date:", feature_date.date())
    print("Weights:", json.dumps(weights))
    print("Selected model:", best_model)
    print("Aggregate forecast:")
    print(final_panel.groupby("target_date")["forecast"].sum().round(1).to_string())
    print("Submitted rows:", len(verify))
    print("Created at:", verify[0].get("created_at") if verify else None)


if __name__ == "__main__":
    run()
