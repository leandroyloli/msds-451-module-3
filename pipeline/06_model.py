"""Choose XGBoost hyperparameters with Optuna on the development window.

One hundred trials, scored by mean ROC AUC under time-series cross-validation.
The hold-out file is not read.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_development() -> tuple[np.ndarray, np.ndarray, list[str]]:
    summary = json.loads(config.SELECTION_SUMMARY_JSON.read_text(encoding="utf-8"))
    features = list(summary["winner"])
    frame = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True).sort("Date")
    return (
        frame.select(features).to_numpy(),
        frame["Target"].to_numpy().astype(np.int32),
        features,
    )


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", *config.XGB_MAX_DEPTH),
        "min_child_weight": trial.suggest_int("min_child_weight", *config.XGB_MIN_CHILD_WEIGHT),
        "subsample": trial.suggest_float("subsample", *config.XGB_SUBSAMPLE),
        "learning_rate": trial.suggest_float("learning_rate", *config.XGB_LEARNING_RATE),
        "n_estimators": trial.suggest_int("n_estimators", *config.XGB_N_ESTIMATORS),
    }


def make_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        **params,
    )


def cv_auc(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=config.N_SPLITS, gap=config.GAP)
    return cross_val_score(model, x, y, cv=splitter, scoring="roc_auc", n_jobs=1)


def _write_chapter(summary: dict) -> None:
    rows = "\n".join(
        f"| {row['rank']} | {row['auc']:.4f} | {row['max_depth']} | {row['min_child_weight']} | "
        f"{row['subsample']:.3f} | {row['learning_rate']:.3f} | {row['n_estimators']} |"
        for row in summary["top"]
    )
    best = summary["best_params"]
    text = f"""# 5. XGBoost

The classifier uses the three columns chosen in the previous chapter: {", ".join(summary["features"])}. Optuna runs {summary["n_trials"]} trials on the {summary["development_rows"]:,} development sessions. Each trial is a different combination of tree depth, minimum child weight, row subsample, learning rate, and number of trees. The next trial is aimed near combinations that already scored well. The score is the mean ROC AUC across {summary["n_splits"]} time-series folds, with a gap of {summary["gap"]} sessions. The subsample is kept between 0.5 and 1. The seed is {summary["random_state"]}. The hold-out file is not read.

The best mean AUC on the development folds is {summary["best_auc"]:.4f}, with a spread across folds of {summary["best_auc_std"]:.4f}. The chosen settings are depth {best["max_depth"]}, minimum child weight {best["min_child_weight"]}, subsample {best["subsample"]:.3f}, learning rate {best["learning_rate"]:.3f}, and {best["n_estimators"]} trees.

## Ten best trials

| Rank | Mean AUC | Depth | Min child | Subsample | Learning rate | Trees |
| --- | --- | --- | --- | --- | --- | --- |
{rows}
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CHAPTER_MD.write_text(text, encoding="utf-8")
    report_spec = importlib.util.spec_from_file_location(
        "report_chapter", Path(__file__).with_name("08_report.py")
    )
    module = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(module)
    pdf = module.render_markdown(
        config.MODEL_CHAPTER_MD,
        config.MODEL_CHAPTER_PDF,
        "MSDS 451 Programming Assignment 1 — XGBoost",
        [0.7 * 72, 0.85 * 72, 0.7 * 72, 0.9 * 72, 0.95 * 72, 1.15 * 72, 0.7 * 72],
    )
    print(f"wrote: {config.MODEL_CHAPTER_MD}")
    print(f"wrote: {pdf}")


def main() -> None:
    x, y, features = load_development()

    def objective(trial: optuna.Trial) -> float:
        scores = cv_auc(make_model(suggest_params(trial)), x, y)
        trial.set_user_attr("auc_std", float(scores.std()))
        return float(scores.mean())

    def _progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        if trial.number % 10 == 0 or trial.number + 1 == config.N_TRIALS:
            print(
                f"trial {trial.number + 1}  auc {trial.value:.4f}  best {study.best_value:.4f}",
                flush=True,
            )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(objective, n_trials=config.N_TRIALS, callbacks=[_progress])

    finished = [trial for trial in study.trials if trial.value is not None]
    finished.sort(key=lambda trial: trial.value, reverse=True)
    top = []
    for rank, trial in enumerate(finished[:10], start=1):
        top.append(
            {
                "rank": rank,
                "auc": trial.value,
                "auc_std": trial.user_attrs.get("auc_std"),
                **trial.params,
            }
        )
    best = study.best_trial
    summary = {
        "features": features,
        "development_rows": int(len(y)),
        "n_trials": config.N_TRIALS,
        "n_splits": config.N_SPLITS,
        "gap": config.GAP,
        "random_state": config.RANDOM_STATE,
        "best_auc": best.value,
        "best_auc_std": best.user_attrs.get("auc_std"),
        "best_params": best.params,
        "top": top,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"best AUC {summary['best_auc']:.4f}  "
        f"depth {best.params['max_depth']}  "
        f"trees {best.params['n_estimators']}  "
        f"learning_rate {best.params['learning_rate']:.3f}"
    )
    print(f"wrote: {config.MODEL_SUMMARY_JSON}")
    _write_chapter(summary)


if __name__ == "__main__":
    main()
