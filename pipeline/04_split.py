"""Cut the feature table into development, gap, and hold-out.

Reads DEV_END and GAP from 01_config.py.
Writes the three CSVs and output/split_summary.json.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import polars as pl

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def split_frame(features: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if config.GAP < 0:
        raise RuntimeError("GAP must be zero or positive")
    dev_end = date.fromisoformat(config.DEV_END)
    ordered = features.sort("Date")
    development = ordered.filter(pl.col("Date") <= dev_end)
    after = ordered.filter(pl.col("Date") > dev_end)
    if development.is_empty():
        raise RuntimeError(f"no sessions on or before DEV_END={config.DEV_END}")
    if after.height < config.GAP:
        raise RuntimeError(
            f"only {after.height} sessions after DEV_END, fewer than GAP={config.GAP}"
        )
    gap = after.head(config.GAP)
    holdout = after.slice(config.GAP)
    if holdout.is_empty():
        raise RuntimeError("hold-out is empty; lower GAP or move DEV_END earlier")
    return development, gap, holdout


def _window(frame: pl.DataFrame) -> dict:
    if frame.is_empty():
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": frame.height,
        "start": str(frame["Date"].min()),
        "end": str(frame["Date"].max()),
    }


def main() -> None:
    features = pl.read_csv(config.FEATURES_CSV, try_parse_dates=True).with_columns(
        pl.col("Date").cast(pl.Date)
    )
    development, gap, holdout = split_frame(features)
    development.write_csv(config.DEVELOPMENT_CSV)
    gap.write_csv(config.GAP_CSV)
    holdout.write_csv(config.HOLDOUT_CSV)
    summary = {
        "dev_end": config.DEV_END,
        "gap_sessions": config.GAP,
        "development": _window(development),
        "gap": _window(gap),
        "holdout": _window(holdout),
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SPLIT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"DEV_END={config.DEV_END}  GAP={config.GAP}")
    print(
        f"development: {summary['development']['rows']}  "
        f"{summary['development']['start']} -> {summary['development']['end']}"
    )
    if config.GAP == 0:
        print("gap: 0")
    else:
        print(
            f"gap: {summary['gap']['rows']}  "
            f"{summary['gap']['start']} -> {summary['gap']['end']}"
        )
    print(
        f"hold-out: {summary['holdout']['rows']}  "
        f"{summary['holdout']['start']} -> {summary['holdout']['end']}"
    )
    print(f"wrote: {config.DEVELOPMENT_CSV}")
    print(f"wrote: {config.HOLDOUT_CSV}")


if __name__ == "__main__":
    main()
