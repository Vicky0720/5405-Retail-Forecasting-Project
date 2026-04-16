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
ORIGINS = [pd.Timestamp("2022-08-04"), pd.Timestamp("2022-08-11"), pd.Timestamp("2022-08-18")]


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

    backtest_rows = []
    backtest_panels = []
    for origin in ORIGINS:
        train_panel = panel[panel["date"] <= origin]
        actual = actual_for(panel, origin)
        merged = merge_candidates(actual, build_candidates(train_panel, origin, holidays))
        merged["origin"] = str(origin.date())
        backtest_panels.append(merged)
        for name in model_names:
            backtest_rows.append(
                {
                    "origin": str(origin.date()),
                    "model": name,
                    "score": weighted_1mape(merged["actual"], merged[f"forecast_{name}"]),
                }
            )

    selection = pd.concat(backtest_panels, ignore_index=True)
    weights = search_weights(selection, model_names)
    ensemble_pred = sum(weights[name] * selection[f"forecast_{name}"].to_numpy(dtype=float) for name in model_names)
    backtest_rows.append({"origin": "ALL", "model": "ensemble", "score": weighted_1mape(selection["actual"], ensemble_pred)})
    for name in model_names:
        backtest_rows.append(
            {
                "origin": "ALL",
                "model": name,
                "score": weighted_1mape(selection["actual"], selection[f"forecast_{name}"]),
            }
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(backtest_rows).to_csv(OUTPUT_DIR / f"course_prophet_pipeline_backtest_{feature_tag}.csv", index=False)
    (OUTPUT_DIR / f"course_prophet_pipeline_weights_{feature_tag}.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )

    candidates = build_candidates(panel, feature_date, holidays)
    final = None
    for name, frame in candidates.items():
        tmp = frame.rename(columns={"forecast": f"forecast_{name}"})
        final = tmp if final is None else final.merge(tmp, on=["option_id", "target_date"], how="inner")
    final["forecast"] = sum(weights[name] * final[f"forecast_{name}"] for name in model_names)
    final_panel = final[["option_id", "target_date", "forecast"]].sort_values(["option_id", "target_date"])
    final_panel.assign(target_date=final_panel["target_date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUTPUT_DIR / f"forecast_option_day_{feature_tag}_prophet_pipeline.csv",
        index=False,
    )

    template = load_template_from_api(api_endpoint, token, scenario_name=DEFAULT_SCENARIO)
    payload = format_submission_payload(final_panel, template=template, horizon=DEFAULT_HORIZON)
    (OUTPUT_DIR / f"forecast_submission_{feature_tag}_prophet_pipeline.json").write_text(
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
    print("Aggregate forecast:")
    print(final_panel.groupby("target_date")["forecast"].sum().round(1).to_string())
    print("Submitted rows:", len(verify))
    print("Created at:", verify[0].get("created_at") if verify else None)


if __name__ == "__main__":
    run()
