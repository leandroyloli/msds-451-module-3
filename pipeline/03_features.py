"""Build the 25 candidate features and the next-session direction target.

Reads data/wege3_sa_prices.csv.
Writes data/wege3_sa_features.csv and output/features_summary.json.

Every candidate on row t is known at the close of t-1.
A recorded volume of zero is treated as missing, not as a trade of zero shares.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import polars as pl

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

_LN2 = math.log(2.0)
_PARKINSON_SCALE = 4.0 * _LN2


def engineer(prices: pl.DataFrame) -> pl.DataFrame:
    frame = prices.sort("Date")
    volume = (
        pl.when(pl.col("Volume") > 0).then(pl.col("Volume")).otherwise(None).alias("_Volume")
    )
    frame = frame.with_columns(volume)

    frame = frame.with_columns(
        pl.col("Close").shift(1).alias("CloseLag1"),
        pl.col("Close").shift(2).alias("CloseLag2"),
        pl.col("Close").shift(3).alias("CloseLag3"),
        (pl.col("High") - pl.col("Low")).alias("HML"),
        (pl.col("Open") - pl.col("Close")).alias("OMC"),
        (pl.col("Close").log() - pl.col("Close").shift(1).log()).alias("_Ret"),
    )
    frame = frame.with_columns(
        pl.col("HML").shift(1).alias("HMLLag1"),
        pl.col("HML").shift(2).alias("HMLLag2"),
        pl.col("HML").shift(3).alias("HMLLag3"),
        pl.col("OMC").shift(1).alias("OMCLag1"),
        pl.col("OMC").shift(2).alias("OMCLag2"),
        pl.col("OMC").shift(3).alias("OMCLag3"),
        pl.col("_Volume").shift(1).alias("VolumeLag1"),
        pl.col("_Volume").shift(2).alias("VolumeLag2"),
        pl.col("_Volume").shift(3).alias("VolumeLag3"),
        pl.col("CloseLag1").ewm_mean(half_life=1, ignore_nulls=True).alias("CloseEMA2"),
        pl.col("CloseLag1").ewm_mean(half_life=2, ignore_nulls=True).alias("CloseEMA4"),
        pl.col("CloseLag1").ewm_mean(half_life=4, ignore_nulls=True).alias("CloseEMA8"),
        pl.col("_Ret").shift(1).alias("Ret1"),
        (pl.col("Close").shift(1).log() - pl.col("Close").shift(22).log()).alias("Ret21"),
        (pl.col("Close").shift(21).log() - pl.col("Close").shift(252).log()).alias("Mom12_1"),
        pl.col("_Ret").shift(1).rolling_std(window_size=21, min_samples=21).alias("RVol21"),
        ((pl.col("High") / pl.col("Low")).log().pow(2) / _PARKINSON_SCALE)
        .shift(1)
        .rolling_mean(window_size=21, min_samples=21)
        .alias("Parkinson21"),
        (pl.col("_Volume").shift(1).log() - pl.col("_Volume").shift(2).log()).alias("LogVolChg1"),
        (pl.col("_Ret").abs() / (pl.col("Close") * pl.col("_Volume")))
        .shift(1)
        .rolling_mean(window_size=21, min_samples=21)
        .alias("Amihud21"),
        (pl.col("HML").shift(1) / pl.col("Close").shift(1)).alias("Range1"),
        (pl.col("Close").shift(1).log() - pl.col("Open").shift(1).log()).alias("Intraday1"),
        (pl.col("Open").shift(1).log() - pl.col("Close").shift(2).log()).alias("Overnight1"),
        pl.col("_Ret").alias("LogReturn"),
    )
    frame = frame.with_columns(
        pl.when(pl.col("LogReturn") > 0.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias("Target")
    )
    _assert_definitions(frame)
    keep = (
        ["Date", "Open", "High", "Low", "Close", "Volume", "HML", "OMC"]
        + config.CANDIDATE_FEATURES
        + ["LogReturn", "Target"]
    )
    model = frame.drop_nulls(subset=config.CANDIDATE_FEATURES + ["LogReturn"]).select(keep)
    return model


def _assert_definitions(frame: pl.DataFrame) -> None:
    """Check formulas on rows where the short-lag features already exist."""
    ready = frame.drop_nulls(subset=["Ret1", "CloseLag1", "CloseLag2", "HMLLag1", "Range1", "Intraday1"])
    gaps = ready.select(
        (pl.col("Ret1") - (pl.col("CloseLag1").log() - pl.col("CloseLag2").log())).abs().max().alias("ret1"),
        (pl.col("Range1") - (pl.col("HMLLag1") / pl.col("CloseLag1"))).abs().max().alias("range1"),
        (pl.col("LogReturn") - (pl.col("Close").log() - pl.col("CloseLag1").log())).abs().max().alias("target_ret"),
    )
    row = gaps.row(0, named=True)
    for name, gap in row.items():
        if gap is None or gap > 1e-8:
            raise RuntimeError(f"feature check failed for {name}: {gap}")


def summarize(prices: pl.DataFrame, features: pl.DataFrame) -> dict:
    up = int(features.filter(pl.col("Target") == 1).height)
    zero_volume = int(prices.filter(pl.col("Volume") == 0).height)
    span = prices.filter(
        (pl.col("Date") >= features["Date"].min()) & (pl.col("Date") <= features["Date"].max())
    )
    return {
        "symbol": config.SYMBOL,
        "price_rows": prices.height,
        "feature_rows": features.height,
        "rows_removed": prices.height - features.height,
        "price_rows_in_span": span.height,
        "removed_inside_span": span.height - features.height,
        "zero_volume_rows": zero_volume,
        "start": str(features["Date"].min()),
        "end": str(features["Date"].max()),
        "up_days": up,
        "down_or_flat_days": features.height - up,
        "up_share": up / features.height,
        "n_features": len(config.CANDIDATE_FEATURES),
        "features": config.CANDIDATE_FEATURES,
    }


def main() -> None:
    prices = pl.read_csv(config.PRICES_CSV, try_parse_dates=True).with_columns(
        pl.col("Date").cast(pl.Date)
    )
    features = engineer(prices)
    features.write_csv(config.FEATURES_CSV)
    summary = summarize(prices, features)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FEATURES_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"rows: {summary['feature_rows']}")
    print(f"from: {summary['start']}")
    print(f"to: {summary['end']}")
    print(f"up days: {summary['up_days']} ({summary['up_share']:.3f})")
    print(f"down or flat: {summary['down_or_flat_days']}")
    print(f"zero-volume sessions in the price file: {summary['zero_volume_rows']}")
    print(f"wrote: {config.FEATURES_CSV}")


if __name__ == "__main__":
    main()
