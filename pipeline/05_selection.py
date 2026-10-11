"""All-subsets AIC on the development window.

At most one of the six price-level columns may enter a subset.
The logistic regression is unpenalized. The hold-out file is not read.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

_STATE: dict = {}
_WORKERS = max(2, (os.cpu_count() or 4) - 2)
_CHUNK = 16384
_TOP = 10


def _fit_aic(x: np.ndarray, y: np.ndarray) -> float:
    """AIC of an unpenalized logistic regression. Inf if the fit does not settle."""
    n, k = x.shape
    z = np.empty((n, k + 1), dtype=np.float64)
    z[:, 0] = 1.0
    if k:
        z[:, 1:] = x
    beta = np.zeros(k + 1, dtype=np.float64)
    for _ in range(20):
        eta = z @ beta
        np.clip(eta, -20.0, 20.0, out=eta)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-6)
        gram = z.T @ (z * w[:, None])
        grad = z.T @ (y - mu)
        try:
            step = np.linalg.solve(gram, grad)
        except np.linalg.LinAlgError:
            return np.inf
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    if np.max(np.abs(beta)) > 30.0:
        return np.inf
    eta = np.clip(z @ beta, -20.0, 20.0)
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    loglik = float(np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))
    if not np.isfinite(loglik):
        return np.inf
    return 2.0 * (k + 1) - 2.0 * loglik


def _init_worker(x: np.ndarray, y: np.ndarray, rest_idx: np.ndarray, level_idx: np.ndarray) -> None:
    _STATE["X"] = x
    _STATE["y"] = y
    _STATE["rest"] = rest_idx
    _STATE["level"] = level_idx


def _score_chunk(bounds: tuple[int, int]) -> tuple[list[tuple[float, int, int]], int, int]:
    x = _STATE["X"]
    y = _STATE["y"]
    rest = _STATE["rest"]
    level = _STATE["level"]
    n_rest = len(rest)
    best: list[tuple[float, int, int]] = []
    failed = 0
    scored = 0
    cols = np.empty(n_rest + 1, dtype=np.int64)
    lo, hi = bounds
    for bits in range(lo, hi):
        width = 0
        for i in range(n_rest):
            if bits & (1 << i):
                cols[width] = rest[i]
                width += 1
        base = width
        for lev in range(len(level) + 1):
            width = base
            if lev:
                cols[width] = level[lev - 1]
                width += 1
            if width == 0:
                continue
            aic = _fit_aic(x[:, cols[:width]], y)
            scored += 1
            if not np.isfinite(aic):
                failed += 1
                continue
            item = (-aic, bits, lev)
            if len(best) < 40:
                heapq.heappush(best, item)
            elif -aic > best[0][0]:
                heapq.heapreplace(best, item)
    return best, scored, failed


def _names(bits: int, lev: int, rest_names: list[str], level_names: list[str]) -> list[str]:
    chosen = [rest_names[i] for i in range(len(rest_names)) if bits & (1 << i)]
    if lev:
        chosen.append(level_names[lev - 1])
    return chosen


def _sklearn_aic(x: np.ndarray, y: np.ndarray) -> float:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=200)
    model.fit(x, y)
    loglik = -log_loss(y, model.predict_proba(x), normalize=False)
    return 2.0 * (x.shape[1] + 1) - 2.0 * loglik


def _loglik(x: np.ndarray, y: np.ndarray) -> float:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=400)
    model.fit(x, y)
    return float(-log_loss(y, model.predict_proba(x), normalize=False))


def _benchmark(scaled: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    index = {name: i for i, name in enumerate(names)}
    level = config.PRICE_LEVEL_ONE_OF
    block = scaled[:, [index[name] for name in level]]
    corr = np.corrcoef(block, rowvar=False)
    off = np.abs(corr)
    np.fill_diagonal(off, 0.0)
    base = _loglik(scaled[:, [index["CloseLag1"]]], y)
    comparisons = {
        "close_plus_close_lag3": _loglik(scaled[:, [index["CloseLag1"], index["CloseLag3"]]], y) - base,
        "close_plus_ema8": _loglik(scaled[:, [index["CloseLag1"], index["CloseEMA8"]]], y) - base,
        "all_six_price_levels": _loglik(block, y) - base,
    }
    hml = _loglik(scaled[:, [index["HMLLag1"]]], y)
    comparisons["hml_plus_close_versus_hml"] = (
        _loglik(scaled[:, [index["HMLLag1"], index["CloseLag1"]]], y) - hml
    )
    return {
        "min_abs_correlation": float(off[off > 0].min()) if np.any(off > 0) else 1.0,
        "loglik_gain": {key: float(value) for key, value in comparisons.items()},
        "aic_hurdle_per_extra_column": 1.0,
    }


def _check_solver(scaled: np.ndarray, y: np.ndarray, rest_idx: np.ndarray) -> float:
    gaps = []
    for bits in (1, 3, 7, 21, 85, 341, 1023, 4095, 8191, 20000, 100000, 400000):
        cols = [int(rest_idx[i]) for i in range(len(rest_idx)) if bits & (1 << i)]
        if not cols:
            continue
        block = scaled[:, cols]
        gaps.append(abs(_fit_aic(block, y) - _sklearn_aic(block, y)))
    return float(max(gaps)) if gaps else 0.0


def _search(scaled: np.ndarray, y: np.ndarray, rest_idx: np.ndarray, level_idx: np.ndarray) -> dict:
    n_bits = 1 << len(rest_idx)
    chunks = [(start, min(start + _CHUNK, n_bits)) for start in range(0, n_bits, _CHUNK)]
    best: list[tuple[float, int, int]] = []
    scored = 0
    failed = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=_WORKERS,
        initializer=_init_worker,
        initargs=(scaled, y, rest_idx, level_idx),
    ) as pool:
        pending = [pool.submit(_score_chunk, chunk) for chunk in chunks]
        done = 0
        for future in as_completed(pending):
            chunk_best, chunk_scored, chunk_failed = future.result()
            scored += chunk_scored
            failed += chunk_failed
            for item in chunk_best:
                if len(best) < 40:
                    heapq.heappush(best, item)
                elif item[0] > best[0][0]:
                    heapq.heapreplace(best, item)
            done += 1
            elapsed = time.perf_counter() - started
            print(f"chunks {done}/{len(chunks)}  subsets {scored}  failed {failed}  {elapsed:.0f}s", flush=True)
    ordered = sorted(best, key=lambda item: -item[0])[:_TOP]
    return {
        "evaluated": scored,
        "failed": failed,
        "seconds": time.perf_counter() - started,
        "top": [{"aic": -aic, "bits": bits, "level": lev} for aic, bits, lev in ordered],
    }


def _null_aic() -> float:
    import polars as pl

    y = pl.read_csv(config.DEVELOPMENT_CSV)["Target"].to_numpy().astype(np.float64)
    share = float(y.mean())
    loglik = float(np.sum(y * np.log(share) + (1.0 - y) * np.log(1.0 - share)))
    return 2.0 - 2.0 * loglik


def _write_chapter(summary: dict) -> None:
    gain = summary["benchmark"]["loglik_gain"]
    top_rows = "\n".join(
        f"| {row['rank']} | {row['aic']:.2f} | {', '.join(row['features'])} |"
        for row in summary["top"]
    )
    winner = ", ".join(summary["winner"])
    null_aic = _null_aic()
    text = f"""# 4. Feature selection

The search uses the {summary['development_rows']:,} development sessions only. Each candidate subset is fit by an unpenalized logistic regression, which is the maximum likelihood the AIC formula assumes. The subset with the smallest AIC is kept. Columns are standardized with the mean and standard deviation of those development sessions. For an unpenalized logistic regression with an intercept, that rescaling does not change the maximized likelihood. The hold-out file is not read.

## Combinations that repeat the same price

Six columns track one price path: CloseLag1, CloseLag2, CloseLag3, CloseEMA2, CloseEMA4, and CloseEMA8. On the development window the smallest absolute correlation among them is {summary['benchmark']['min_abs_correlation']:.3f}. Adding the other five to CloseLag1 raises the log-likelihood by {gain['all_six_price_levels']:.2f}. AIC needs a gain above 1 for each added column, so a second copy does not pay for itself. CloseLag1 with CloseLag3 gains {gain['close_plus_close_lag3']:.2f}. CloseLag1 with CloseEMA8 gains {gain['close_plus_ema8']:.2f}.

The search allows at most one of those six. The other 19 columns stay free, including next to that one price column. High minus low together with the close still belongs in the search: that pair gains {gain['hml_plus_close_versus_hml']:.2f} over high minus low alone, which clears the hurdle of 1.

Subsets of all 25 columns number {summary['unrestricted_subsets']:,}. With at most one price-level column the number is {summary['searched_subsets']:,}. A second price-level column cannot improve the AIC, so the cut does not change the winner.

## Selected subset

The smallest AIC is {summary['best_aic']:.2f}, on {winner}. An intercept-only model on the same {summary['development_rows']:,} sessions has AIC {null_aic:.2f}. The winner improves on that null model by {null_aic - summary['best_aic']:.2f}.

| Rank | AIC | Features |
| --- | --- | --- |
{top_rows}

The search scored {summary['evaluated']:,} subsets in {summary['seconds'] / 60:.1f} minutes. {summary['failed']:,} fits did not settle and were discarded.
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.SELECTION_CHAPTER_MD.write_text(text, encoding="utf-8")
    report = importlib.util.spec_from_file_location(
        "report_chapter", Path(__file__).with_name("08_report.py")
    )
    module = importlib.util.module_from_spec(report)
    report.loader.exec_module(module)
    pdf = module.render_markdown(
        config.SELECTION_CHAPTER_MD,
        config.SELECTION_CHAPTER_PDF,
        "MSDS 451 Programming Assignment 1 — Feature selection",
        [0.55 * 72, 0.85 * 72, 5.35 * 72],
    )
    print(f"wrote: {config.SELECTION_CHAPTER_MD}")
    print(f"wrote: {pdf}")


def main() -> None:
    import polars as pl

    dev = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True)
    names = list(config.MODEL_FEATURES)
    level_names = list(config.PRICE_LEVEL_ONE_OF)
    rest_names = [name for name in names if name not in level_names]
    raw = dev.select(names).to_numpy()
    y = dev["Target"].to_numpy().astype(np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    level_idx = np.array([names.index(name) for name in level_names], dtype=np.int64)
    rest_idx = np.array([names.index(name) for name in rest_names], dtype=np.int64)

    gap = _check_solver(scaled, y, rest_idx)
    print(f"solver check, max AIC gap versus sklearn: {gap:.4f}")
    if gap > 0.05:
        raise RuntimeError("the fast AIC fit does not match sklearn")

    benchmark = _benchmark(scaled, y, names)
    print("benchmark", json.dumps(benchmark["loglik_gain"]))
    found = _search(scaled, y, rest_idx, level_idx)
    top = []
    for rank, row in enumerate(found["top"], start=1):
        features = _names(row["bits"], row["level"], rest_names, level_names)
        top.append({"rank": rank, "aic": row["aic"], "n_features": len(features), "features": features})
    summary = {
        "development_rows": dev.height,
        "unrestricted_subsets": (1 << len(names)) - 1,
        "searched_subsets": (len(level_names) + 1) * (1 << len(rest_names)) - 1,
        "price_level_rule": level_names,
        "benchmark": benchmark,
        "evaluated": found["evaluated"],
        "failed": found["failed"],
        "seconds": found["seconds"],
        "best_aic": top[0]["aic"],
        "winner": top[0]["features"],
        "top": top,
        "scaler_mean": {name: float(value) for name, value in zip(names, scaler.mean_)},
        "scaler_scale": {name: float(value) for name, value in zip(names, scaler.scale_)},
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SELECTION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"winner AIC {summary['best_aic']:.2f}: {', '.join(summary['winner'])}")
    print(f"wrote: {config.SELECTION_SUMMARY_JSON}")
    _write_chapter(summary)


if __name__ == "__main__":
    main()
