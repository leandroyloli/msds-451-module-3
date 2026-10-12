"""Choose XGBoost hyperparameters with Optuna on the development window.

One hundred trials, scored by mean ROC AUC under time-series cross-validation.
The chosen trial is the simplest one within one standard error of the best,
not the best itself. The hold-out file is not read.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_development() -> tuple[np.ndarray, np.ndarray, list[str]]:
    summary = json.loads(config.SELECTION_SUMMARY_JSON.read_text(encoding="utf-8"))
    features = list(summary["final"])
    frame = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True).sort("Date")
    return (
        frame.select(features).to_numpy(),
        frame["Target"].to_numpy().astype(np.int32),
        features,
    )


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", *config.XGB_MAX_DEPTH),
        "min_child_weight": trial.suggest_int("min_child_weight", *config.XGB_MIN_CHILD_WEIGHT, log=True),
        "subsample": trial.suggest_float("subsample", *config.XGB_SUBSAMPLE),
        "learning_rate": trial.suggest_float("learning_rate", *config.XGB_LEARNING_RATE, log=True),
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


def logistic() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=1000))


def cv_auc(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=config.N_SPLITS, gap=config.GAP)
    return cross_val_score(model, x, y, cv=splitter, scoring="roc_auc", n_jobs=1)


def one_standard_error(trials: list) -> optuna.trial.FrozenTrial:
    """Simplest trial whose mean AUC is within one standard error of the best.

    Simpler means shallower trees first, then fewer trees.
    """
    best = max(trials, key=lambda trial: trial.value)
    se = best.user_attrs["auc_std"] / np.sqrt(config.N_SPLITS)
    close = [trial for trial in trials if trial.value >= best.value - se]
    return min(close, key=lambda trial: (trial.params["max_depth"], trial.params["n_estimators"]))


def _write_chapter(summary: dict) -> None:
    rows = "\n".join(
        f"| {row['rank']} | {row['auc']:.4f} | {row['max_depth']} | {row['min_child_weight']} | "
        f"{row['subsample']:.3f} | {row['learning_rate']:.3f} | {row['n_estimators']} |"
        for row in summary["top"]
    )
    best = summary["best_params"]
    top = summary["top_trial"]
    ranges = summary["ranges"]
    text = f"""# 6. XGBoost

The classifier uses the {len(summary["features"])} columns passed on by the previous chapter: {", ".join(summary["features"])}. Optuna runs {summary["n_trials"]} trials on the {summary["development_rows"]:,} development sessions. Each trial is a different combination of tree depth, minimum child weight, row subsample, learning rate, and number of trees. The next trial is aimed near combinations that already scored well. The score is the mean ROC AUC across {summary["n_splits"]} time-series folds, with a gap of {summary["gap"]} sessions:

$$\\overline{{\\mathrm{{AUC}}}} = \\frac{{1}}{{{summary["n_splits"]}}}\\sum_{{k=1}}^{{{summary["n_splits"]}}}\\mathrm{{AUC}}_k.$$

Tree depth runs from {ranges["max_depth"][0]} to {ranges["max_depth"][1]}. The subsample is kept between {ranges["subsample"][0]:g} and {ranges["subsample"][1]:g}. The seed is {summary["random_state"]}. The hold-out file is not read.

Many trials score almost the same, so the single best one is partly luck. The search keeps the simplest trial whose score is within one standard error of the best: shallower trees first, then fewer trees. The best mean AUC is {top["auc"]:.4f}, and {summary["band_size"]} of the {summary["n_trials"]} trials fall within one standard error of it.

The chosen settings are depth {best["max_depth"]}, minimum child weight {best["min_child_weight"]}, subsample {best["subsample"]:.3f}, learning rate {best["learning_rate"]:.3f}, and {best["n_estimators"]} trees. Their mean AUC is {summary["best_auc"]:.4f}. A logistic regression on the same columns and folds reaches {summary["logistic_auc"]:.4f}.

## Ten best trials

| Rank | Mean AUC | Depth | Min child | Subsample | Learning rate | Trees |
| --- | --- | --- | --- | --- | --- | --- |
{rows}
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_CHAPTER_MD.write_text(text, encoding="utf-8")
    print(f"wrote: {config.MODEL_CHAPTER_MD}")


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
    chosen = one_standard_error(finished)
    best = finished[0]
    floor = best.value - best.user_attrs["auc_std"] / np.sqrt(config.N_SPLITS)
    top = [
        {"rank": rank, "auc": trial.value, "auc_std": trial.user_attrs.get("auc_std"), **trial.params}
        for rank, trial in enumerate(finished[:10], start=1)
    ]
    logistic_scores = cv_auc(logistic(), x, y)
    summary = {
        "features": features,
        "development_rows": int(len(y)),
        "n_trials": config.N_TRIALS,
        "n_splits": config.N_SPLITS,
        "gap": config.GAP,
        "random_state": config.RANDOM_STATE,
        "ranges": {
            "max_depth": list(config.XGB_MAX_DEPTH),
            "min_child_weight": list(config.XGB_MIN_CHILD_WEIGHT),
            "subsample": list(config.XGB_SUBSAMPLE),
            "learning_rate": list(config.XGB_LEARNING_RATE),
            "n_estimators": list(config.XGB_N_ESTIMATORS),
        },
        "top_trial": {"auc": best.value, "auc_std": best.user_attrs["auc_std"], **best.params},
        "band_floor": float(floor),
        "band_size": sum(1 for trial in finished if trial.value >= floor),
        "chosen_rank": finished.index(chosen) + 1,
        "best_auc": chosen.value,
        "best_auc_std": chosen.user_attrs.get("auc_std"),
        "best_params": chosen.params,
        "logistic_auc": float(logistic_scores.mean()),
        "logistic_auc_std": float(logistic_scores.std()),
        "top": top,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"chosen AUC {summary['best_auc']:.4f} (rank {summary['chosen_rank']})  "
        f"depth {chosen.params['max_depth']}  "
        f"trees {chosen.params['n_estimators']}  "
        f"learning_rate {chosen.params['learning_rate']:.3f}  "
        f"logistic {summary['logistic_auc']:.4f}"
    )
    print(f"wrote: {config.MODEL_SUMMARY_JSON}")
    _write_chapter(summary)


if __name__ == "__main__":
    main()
