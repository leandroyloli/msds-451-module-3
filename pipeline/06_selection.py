"""Forward selection on the observable universe, with a noise gate.

A subset is scored by out-of-sample log loss under time-series cross-validation.
The search starts from no column and adds the column that lowers that loss most,
until no column lowers it. The gain over the no-column forecast is kept only when
it beats the gains the same search finds after the label is rolled against the
features. The procedure is also nested inside an outer time-series split, so the
development estimate does not reuse the sessions that chose the columns.

Reads data/wege3_sa_development.csv and output/universe_summary.json.
The hold-out file is not read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

_WORKERS = max(2, (os.cpu_count() or 4) - 2)
_EPS = 1e-12


def fit_logit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logistic regression by Newton steps on columns standardized with the training rows."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = np.zeros(z.shape[1])
    ridge = 1e-8 * np.eye(z.shape[1])
    for _ in range(25):
        mu = 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -30.0, 30.0)))
        w = np.maximum(mu * (1.0 - mu), 1e-9)
        step = np.linalg.solve(z.T @ (z * w[:, None]) + ridge, z.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta, mean, scale


def predict_logit(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    beta, mean, scale = model
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    return 1.0 / (1.0 + np.exp(-np.clip(z @ beta, -30.0, 30.0)))


def _loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(TimeSeriesSplit(n_splits=config.N_SPLITS, gap=config.GAP).split(np.arange(n)))


def cv_loss(x: np.ndarray, y: np.ndarray, cols: list[int], splits: list) -> float:
    """Mean out-of-sample log loss. With no column the forecast is the training up share."""
    losses = []
    for train, test in splits:
        if cols:
            model = fit_logit(x[np.ix_(train, cols)], y[train])
            p = predict_logit(model, x[np.ix_(test, cols)])
        else:
            p = np.full(len(test), y[train].mean())
        losses.append(_loss(y[test], p))
    return float(np.mean(losses))


def forward(x: np.ndarray, y: np.ndarray, splits: list) -> dict:
    chosen: list[int] = []
    base = cv_loss(x, y, chosen, splits)
    current = base
    path = []
    while len(chosen) < x.shape[1]:
        trial = [(cv_loss(x, y, chosen + [j], splits), j) for j in range(x.shape[1]) if j not in chosen]
        loss, j = min(trial)
        if loss >= current:
            break
        chosen.append(j)
        current = loss
        path.append((j, loss))
    return {"chosen": chosen, "base_loss": base, "loss": current, "gain": base - current, "path": path}


def _shift_offsets(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(config.MIN_SHIFT, n - config.MIN_SHIFT, size=count)


def _shifted_gain(args: tuple[np.ndarray, np.ndarray, int]) -> float:
    x, y, offset = args
    return forward(x, np.roll(y, offset), folds(len(y)))["gain"]


def gated_forward(x: np.ndarray, y: np.ndarray, pool: ProcessPoolExecutor, seed: int) -> dict:
    """Forward selection plus the noise gate. Returns the chosen columns and the gate result."""
    real = forward(x, y, folds(len(y)))
    offsets = _shift_offsets(len(y), config.N_SHIFTS, seed)
    noise = np.array(list(pool.map(_shifted_gain, [(x, y, int(k)) for k in offsets], chunksize=4)))
    threshold = float(np.quantile(noise, config.GATE_QUANTILE))
    passed = bool(real["chosen"]) and real["gain"] > threshold
    return {
        **real,
        "noise_gains": noise,
        "threshold": threshold,
        "p_value": float((1 + np.sum(noise >= real["gain"])) / (1 + len(noise))),
        "passed": passed,
        "final": real["chosen"] if passed else list(range(x.shape[1])),
    }


def nested(x: np.ndarray, y: np.ndarray, dates: list, pool: ProcessPoolExecutor, names: list[str]) -> list[dict]:
    """Repeat the gated selection on each outer training window and score the next window."""
    rows = []
    for k, (train, test) in enumerate(folds(len(y))):
        inner = gated_forward(x[train], y[train], pool, config.RANDOM_STATE + k + 1)
        cols = inner["final"]
        model = fit_logit(x[np.ix_(train, cols)], y[train])
        p = predict_logit(model, x[np.ix_(test, cols)])
        base = np.full(len(test), y[train].mean())
        rows.append(
            {
                "fold": k + 1,
                "train_rows": int(len(train)),
                "test_start": str(dates[int(test[0])]),
                "test_end": str(dates[int(test[-1])]),
                "passed": inner["passed"],
                "chosen": [names[j] for j in inner["chosen"]],
                "used": [names[j] for j in cols],
                "auc": float(roc_auc_score(y[test], p)),
                "loss": _loss(y[test], p),
                "base_loss": _loss(y[test], base),
            }
        )
        print(
            f"outer fold {k + 1}: passed={inner['passed']} chosen={rows[-1]['chosen']} "
            f"auc={rows[-1]['auc']:.4f}",
            flush=True,
        )
    return rows


def _check_solver(x: np.ndarray, y: np.ndarray) -> float:
    """Largest gap in predicted probability against scikit-learn's unpenalized fit."""
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-12)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    model.fit(z, y)
    ours = predict_logit(fit_logit(x, y), x)
    gap = float(np.max(np.abs(ours - model.predict_proba(z)[:, 1])))
    check = abs(_loss(y, ours) - log_loss(y, model.predict_proba(z)[:, 1]))
    return max(gap, check)


def _write_chapter(summary: dict) -> None:
    path_rows = "\n".join(
        f"| {step['step']} | {step['added']} | {step['loss']:.5f} | {1e3 * (summary['base_loss'] - step['loss']):.3f} |"
        for step in summary["path"]
    )
    if not path_rows:
        path_rows = "| 1 | none | | |"
    nested_rows = "\n".join(
        f"| {row['fold']} | {row['test_start']} to {row['test_end']} | "
        f"{', '.join(row['chosen']) if row['chosen'] else 'none'} | {'Yes' if row['passed'] else 'No'} | "
        f"{', '.join(row['used']) if row['passed'] else 'all ' + str(len(row['used']))} | {row['auc']:.4f} | "
        f"{1e3 * (row['base_loss'] - row['loss']):.3f} |"
        for row in summary["nested"]
    )
    gate = summary["gate"]
    gain = 1e3 * summary["gain"]
    line = 1e3 * gate["threshold"]
    share = 100 * gate["p_value"]
    chosen = ", ".join(summary["chosen"]) if summary["chosen"] else "no column"
    if summary["passed"]:
        verdict = (
            f"The actual gain, {gain:.3f}, is above that line. Only {share:.0f} percent of the shifted runs reach it. "
            f"The next chapter fits XGBoost on {chosen}."
        )
    else:
        verdict = (
            f"The actual gain, {gain:.3f}, is below that line. {share:.0f} percent of the shifted runs reach a gain at least "
            f"as large. The search therefore does not single out any column, and the next chapter fits XGBoost on all "
            f"{len(summary['final'])} columns."
        )
    passed_outer = sum(1 for row in summary["nested"] if row["passed"])
    text = f"""# 5. Feature selection

Selection uses the {len(summary['universe'])} columns from the previous chapter and the {summary['development_rows']:,} development sessions. The hold-out file is not read.

## How a subset is scored

Each subset is fit by logistic regression and scored on sessions the fit did not see. Development is cut into the same {summary['n_splits']} time-series folds used later for XGBoost, with a gap of {summary['gap']} sessions. In each fold the model is fit on earlier sessions and scored on the next block. The score is the log loss,

$$\\ell = -\\frac{{1}}{{n}}\\sum_t \\bigl[y_t \\ln \\hat p_t + (1 - y_t) \\ln(1 - \\hat p_t)\\bigr],$$

averaged over the {summary['n_splits']} folds. Lower is better. A model with no column predicts the up share of its training block every day. Its log loss, {summary['base_loss']:.5f}, is the baseline.

## Forward search

The search starts with no column. At each step it adds the one column that lowers the log loss most. It stops when no column lowers it further. The gain is how far the log loss falls below the baseline, in thousandths.

| Step | Added | Mean log loss | Gain (×1000) |
| --- | --- | --- | --- |
{path_rows}

The search stops at {chosen}, with a gain of {gain:.3f}.

## Noise gate

A search like this always finds some gain, even when the columns carry no information. The gate asks how large a gain noise alone produces.

The label column is shifted in time by at least {gate['min_shift']} sessions, so each day's features line up with the label of another day. The forward search is then run again on the shifted label. This is repeated {gate['n_shifts']} times. The gains from these runs are what noise alone produces. Their median is {1e3 * gate['median']:.3f}, and {100 * gate['quantile']:g} percent of them are below {line:.3f}. A selection is kept only if its gain is above that line.

{verdict}

## A fair estimate

The columns above were chosen with the whole development window. Scoring them on the same window would be too generous. To get a fair estimate, the full procedure, search and gate, is repeated inside each fold using only that fold's training sessions. When the gate fails, the fold uses all {len(summary['universe'])} columns. The resulting logistic regression is scored on the next block, which the procedure did not see.

| Fold | Scored block | Forward pick | Gate passed | Columns used | ROC AUC | Gain (×1000) |
| --- | --- | --- | --- | --- | --- | --- |
{nested_rows}

The gate passes in {passed_outer} of {len(summary['nested'])} folds. The mean ROC AUC is {summary['nested_auc']:.4f}, and the mean gain is {1e3 * summary['nested_gain']:.3f}. A negative gain means the model did worse than predicting the training up share.
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.SELECTION_CHAPTER_MD.write_text(text, encoding="utf-8")
    print(f"wrote: {config.SELECTION_CHAPTER_MD}")


def main() -> None:
    universe = json.loads(config.UNIVERSE_SUMMARY_JSON.read_text(encoding="utf-8"))["universe"]
    dev = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True).sort("Date")
    x = dev.select(universe).to_numpy().astype(np.float64)
    y = dev["Target"].to_numpy().astype(np.float64)
    dates = [str(d) for d in dev["Date"].to_list()]

    gap = _check_solver(x, y)
    print(f"solver check, max gap versus sklearn: {gap:.2e}")
    if gap > 1e-3:
        raise RuntimeError("the logistic fit does not match sklearn")

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
        result = gated_forward(x, y, pool, config.RANDOM_STATE)
        print(
            f"full development: chosen={[universe[j] for j in result['chosen']]} gain={result['gain']:.6f} "
            f"threshold={result['threshold']:.6f} passed={result['passed']}",
            flush=True,
        )
        outer = nested(x, y, dates, pool, universe)
    seconds = time.perf_counter() - started

    summary = {
        "development_rows": dev.height,
        "universe": universe,
        "n_splits": config.N_SPLITS,
        "gap": config.GAP,
        "base_loss": result["base_loss"],
        "loss": result["loss"],
        "gain": result["gain"],
        "path": [
            {"step": i + 1, "added": universe[j], "loss": loss} for i, (j, loss) in enumerate(result["path"])
        ],
        "chosen": [universe[j] for j in result["chosen"]],
        "passed": result["passed"],
        "final": [universe[j] for j in result["final"]],
        "gate": {
            "n_shifts": config.N_SHIFTS,
            "min_shift": config.MIN_SHIFT,
            "quantile": config.GATE_QUANTILE,
            "threshold": result["threshold"],
            "median": float(np.median(result["noise_gains"])),
            "p_value": result["p_value"],
        },
        "nested": outer,
        "nested_auc": float(np.mean([row["auc"] for row in outer])),
        "nested_gain": float(np.mean([row["base_loss"] - row["loss"] for row in outer])),
        "seconds": seconds,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.SELECTION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"final columns: {', '.join(summary['final'])}  ({seconds / 60:.1f} minutes)")
    print(f"wrote: {config.SELECTION_SUMMARY_JSON}")
    _write_chapter(summary)


if __name__ == "__main__":
    main()
