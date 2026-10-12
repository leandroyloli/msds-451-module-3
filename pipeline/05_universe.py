"""Screen the 25 candidates for drift on the development window.

A model fit on earlier sessions is applied to later ones. A column whose later
values leave the range seen earlier cannot be used that way: a tree sends every
such value to its outermost leaf. The screen compares the first and the last
third of development and keeps the columns that stay in range.

Reads data/wege3_sa_development.csv. The hold-out file is not read.
Writes output/universe_summary.json and report/04-universe.md.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def screen(development: pl.DataFrame) -> dict:
    ordered = development.sort("Date")
    third = ordered.height // 3
    first = ordered.head(third)
    last = ordered.tail(third)
    low_q, high_q = config.UNIVERSE_QUANTILES
    rows = []
    for name in config.CANDIDATE_FEATURES:
        early = first[name].to_numpy()
        late = last[name].to_numpy()
        low, high = np.quantile(early, [low_q, high_q])
        outside = float(np.mean((late < low) | (late > high)))
        rows.append(
            {
                "feature": name,
                "first_median": float(np.median(early)),
                "last_median": float(np.median(late)),
                "first_low": float(low),
                "first_high": float(high),
                "outside_share": outside,
                "kept": outside <= config.UNIVERSE_MAX_OUTSIDE,
            }
        )
    universe = [row["feature"] for row in rows if row["kept"]]
    if not universe:
        raise RuntimeError("no candidate passes the drift screen")
    return {
        "development_rows": ordered.height,
        "third_rows": third,
        "first_window": [str(first["Date"][0]), str(first["Date"][-1])],
        "last_window": [str(last["Date"][0]), str(last["Date"][-1])],
        "quantiles": list(config.UNIVERSE_QUANTILES),
        "max_outside": config.UNIVERSE_MAX_OUTSIDE,
        "rows": rows,
        "universe": universe,
        "excluded": [row["feature"] for row in rows if not row["kept"]],
    }


def _number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{value:.4g}"


def _ordinal(value: float) -> str:
    number = int(round(value))
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _write_chapter(summary: dict) -> None:
    rows = "\n".join(
        f"| {row['feature']} | {_number(row['first_median'])} | {_number(row['last_median'])} | "
        f"{100 * row['outside_share']:.1f}% | {'Kept' if row['kept'] else 'Excluded'} |"
        for row in summary["rows"]
    )
    first_start, first_end = summary["first_window"]
    last_start, last_end = summary["last_window"]
    low_q, high_q = summary["quantiles"]
    limit = 100 * summary["max_outside"]
    kept = summary["universe"]
    excluded = summary["excluded"]
    text = f"""# 4. Observable universe

A classifier fit on earlier sessions is applied to later ones. That works only for a column whose later values fall in the range seen earlier. A tree cannot extrapolate: a value above everything in the training rows lands in the same leaf as the largest training value. A column measured in reais or in shares grows with the company, so its later values leave the earlier range and the column marks the calendar period instead of the next session.

The screen reads the development file only. It splits development into thirds of {summary['third_rows']:,} sessions. The first third runs from {first_start} through {first_end}. The last third runs from {last_start} through {last_end}. For each candidate $x$, let $q_{{{low_q:g}}}$ and $q_{{{high_q:g}}}$ be the {_ordinal(100 * low_q)} and {_ordinal(100 * high_q)} percentiles over the first third. The outside share is

$$s = \\frac{{1}}{{n_3}}\\sum_{{t \\in \\text{{last third}}}} \\mathbf{{1}}\\{{x_t < q_{{{low_q:g}}} \\ \\text{{or}}\\ x_t > q_{{{high_q:g}}}\\}}.$$

A column with a stable distribution leaves about {100 * (low_q + 1 - high_q):g} percent of the last third outside that range. A candidate enters the observable universe when $s \\le {summary['max_outside']:g}$, five times that rate.

| Feature | First-third median | Last-third median | Outside share | Screen |
| --- | --- | --- | --- | --- |
{rows}

{len(kept)} of {len(summary['rows'])} candidates pass: {', '.join(kept)}. The {len(excluded)} excluded columns are {', '.join(excluded)}. Every excluded column is measured in reais, in shares, or, for Amihud21, per real traded. Each scale-free counterpart that measures the same quantity passes the screen: Ret1 for the lagged close, Range1 for high minus low, Intraday1 for open minus close, and LogVolChg1 for volume. The limit of {limit:g} percent and the two percentiles are the only settings of the screen.
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.UNIVERSE_CHAPTER_MD.write_text(text, encoding="utf-8")
    print(f"wrote: {config.UNIVERSE_CHAPTER_MD}")


def main() -> None:
    development = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True)
    summary = screen(development)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.UNIVERSE_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in summary["rows"]:
        flag = "kept" if row["kept"] else "excluded"
        print(f"{row['feature']:12s} outside {row['outside_share']:6.1%}  {flag}")
    print(f"universe ({len(summary['universe'])}): {', '.join(summary['universe'])}")
    print(f"wrote: {config.UNIVERSE_SUMMARY_JSON}")
    _write_chapter(summary)


if __name__ == "__main__":
    main()
