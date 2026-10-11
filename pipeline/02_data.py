"""Download daily WEGE3.SA prices from Yahoo Finance and save a CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import yfinance as yf

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def download_prices() -> pl.DataFrame:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = yf.Ticker(config.SYMBOL).history(
        start=config.START,
        end=config.END,
        auto_adjust=True,
    )
    if frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {config.SYMBOL}")

    frame = frame.reset_index()
    frame.to_csv(config.PRICES_CSV, index=False)

    prices = pl.read_csv(config.PRICES_CSV, try_parse_dates=True)
    date_col = "Date" if "Date" in prices.columns else prices.columns[0]
    if date_col != "Date":
        prices = prices.rename({date_col: "Date"})
    drop_cols = [
        name
        for name in ("Dividends", "Stock Splits", "StockSplits", "Capital Gains")
        if name in prices.columns
    ]
    prices = prices.drop(drop_cols)
    prices = prices.with_columns(pl.col("Date").cast(pl.Date)).sort("Date")
    prices.write_csv(config.PRICES_CSV)
    return prices


def main() -> None:
    prices = download_prices()
    print(f"symbol: {config.SYMBOL}")
    print(f"rows: {prices.height}")
    print(f"from: {prices['Date'].min()}")
    print(f"to: {prices['Date'].max()}")
    print(f"columns: {prices.columns}")
    print(f"wrote: {config.PRICES_CSV}")


if __name__ == "__main__":
    main()
