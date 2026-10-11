"""Fit the chosen XGBoost once and score the hold-out once.

Reads the features and hyperparameters already stored by 06_model.py.
Does not rerun Optuna and does not change the chosen model.
"""

from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def _iso(value: object) -> str:
    return str(value)[:10]


def _load_model_summary() -> dict:
    summary = json.loads(config.MODEL_SUMMARY_JSON.read_text(encoding="utf-8"))
    params = summary["best_params"]
    summary["best_params"] = {
        "max_depth": int(params["max_depth"]),
        "min_child_weight": int(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
    }
    summary["features"] = list(summary["features"])
    return summary


def _previous_direction(holdout: pl.DataFrame) -> np.ndarray:
    history = (
        pl.read_csv(config.FEATURES_CSV, try_parse_dates=True)
        .sort("Date")
        .select(["Date", pl.col("Target").shift(1).alias("PrevTarget")])
    )
    joined = holdout.join(history, on="Date", how="left")
    if joined["PrevTarget"].null_count():
        raise RuntimeError("a hold-out row has no previous session")
    if config.GAP_CSV.exists():
        gap = pl.read_csv(config.GAP_CSV, try_parse_dates=True).sort("Date")
        if gap.height:
            expected = int(gap["Target"][-1])
            got = int(joined["PrevTarget"][0])
            if expected != got:
                raise RuntimeError("first hold-out day does not follow the last gap day")
    return joined["PrevTarget"].to_numpy().astype(np.int32)


def _fit(summary: dict, development: pl.DataFrame) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        **summary["best_params"],
    )
    model.fit(
        development.select(summary["features"]).to_numpy(),
        development["Target"].to_numpy().astype(np.int32),
    )
    return model


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = float(roc_auc_score(y_true, y_score))
        except ValueError:
            value = float("nan")
    if math.isnan(value):
        return 0.5
    return value


def _rule(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    down_hit, down_miss, up_miss, up_hit = (int(value) for value in matrix.ravel())
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["down or flat", "up"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": _auc(y_true, y_score),
        "down_hit": down_hit,
        "down_miss": down_miss,
        "up_miss": up_miss,
        "up_hit": up_hit,
        "report": report,
        "matrix": matrix.tolist(),
    }


def _save_figures(y_true: np.ndarray, probability: np.ndarray, rules: list[dict]) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probability)
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    axis.plot(fpr, tpr, color="black", label=f"XGBoost {rules[0]['auc']:.3f}")
    axis.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="Chance")
    for rule, marker in zip(rules[1:], ("o", "s")):
        actual_down = rule["down_hit"] + rule["down_miss"]
        actual_up = rule["up_hit"] + rule["up_miss"]
        axis.scatter(
            rule["down_miss"] / actual_down,
            rule["up_hit"] / actual_up,
            marker=marker,
            label=rule["name"],
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Hold-out ROC")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(config.ROC_FIGURE, dpi=150)
    plt.close(figure)

    matrix = np.array(rules[0]["matrix"])
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Down or flat", "Up"])
    axis.set_yticks([0, 1], ["Down or flat", "Up"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Hold-out confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(config.CONFUSION_FIGURE, dpi=150)
    plt.close(figure)


def _write_text(rules: list[dict]) -> None:
    lines = []
    for rule in rules:
        lines.append(rule["name"])
        lines.append("Confusion Matrix")
        lines.append(str(np.array(rule["matrix"])))
        lines.append(_report_text(rule))
        lines.append("")
    config.HOLDOUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def _report_text(rule: dict) -> str:
    report = rule["report"]
    header = f"{'':16} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    rows = [header]
    for label in ("down or flat", "up"):
        item = report[label]
        rows.append(
            f"{label:16} {item['precision']:10.4f} {item['recall']:10.4f} "
            f"{item['f1-score']:10.4f} {int(item['support']):10d}"
        )
    rows.append("")
    rows.append(f"{'accuracy':16} {'':>10} {'':>10} {report['accuracy']:10.4f} {rule['down_hit'] + rule['down_miss'] + rule['up_miss'] + rule['up_hit']:10d}")
    for label in ("macro avg", "weighted avg"):
        item = report[label]
        rows.append(
            f"{label:16} {item['precision']:10.4f} {item['recall']:10.4f} "
            f"{item['f1-score']:10.4f} {int(item['support']):10d}"
        )
    return "\n".join(rows)


def _write_chapter(summary: dict, rules: list[dict], windows: dict) -> None:
    model, always, repeat = rules
    rows = "\n".join(
        f"| {rule['name']} | {rule['accuracy']:.4f} | {rule['auc']:.4f} | "
        f"{rule['down_hit']} | {rule['down_miss']} | {rule['up_miss']} | {rule['up_hit']} |"
        for rule in rules
    )
    best = summary["best_params"]
    text = f"""# 6. Hold-out

The classifier is fit once on the {windows["development_rows"]:,} development sessions, from {windows["development_start"]} through {windows["development_end"]}. The settings are the ones chosen in the previous chapter: depth {best["max_depth"]}, minimum child weight {best["min_child_weight"]}, subsample {best["subsample"]:.3f}, learning rate {best["learning_rate"]:.3f}, and {best["n_estimators"]} trees. The columns are {", ".join(summary["features"])}. The seed is {summary["random_state"]}. The fit does not see the hold-out.

The score uses the {windows["holdout_rows"]:,} hold-out sessions, from {windows["holdout_start"]} through {windows["holdout_end"]}. The {windows["gap_rows"]} sessions from {windows["gap_start"]} through {windows["gap_end"]} stay out of the fit and out of the score. A session is called up when the fitted probability is at least 0.5. The development cross-validation AUC was {summary["best_auc"]:.4f}. That number is not recomputed here.

Two reference rules use no fitted parameters. Always up predicts an up day every session. Repeat previous direction predicts that today's direction equals the direction of the immediately previous session. For the first hold-out day, that previous session is {windows["gap_end"]}.

Down hit counts down-or-flat days predicted as down-or-flat. Down miss counts those days predicted as up. Up miss counts up days predicted as down-or-flat. Up hit counts up days predicted as up. The always-up rule assigns the same score to every session, so its ROC AUC is 0.5.

## Hold-out scores

| Rule | Accuracy | ROC AUC | Down hit | Down miss | Up miss | Up hit |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

The hold-out share of up days is {windows["up_share"]:.4f}. Always up is therefore right on {always["accuracy"]:.4f} of the hold-out sessions. The fitted classifier is right on {model["accuracy"]:.4f}, with ROC AUC {model["auc"]:.4f}. Repeat previous direction is right on {repeat["accuracy"]:.4f}, with ROC AUC {repeat["auc"]:.4f}.

![Figure 1. Hold-out ROC curve. The dashed line is chance.](../figures/holdout_roc.png)

![Figure 2. Hold-out confusion matrix for the fitted classifier.](../figures/holdout_confusion.png)
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_CHAPTER_MD.write_text(text, encoding="utf-8")
    report_spec = importlib.util.spec_from_file_location(
        "report_chapter", Path(__file__).with_name("08_report.py")
    )
    module = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(module)
    pdf = module.render_markdown(
        config.EVAL_CHAPTER_MD,
        config.EVAL_CHAPTER_PDF,
        "MSDS 451 Programming Assignment 1 — Hold-out",
        [1.95 * 72, 0.95 * 72, 0.85 * 72, 0.80 * 72, 0.85 * 72, 0.80 * 72, 0.80 * 72],
    )
    print(f"wrote: {config.EVAL_CHAPTER_MD}")
    print(f"wrote: {pdf}")


def main() -> None:
    summary = _load_model_summary()
    development = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True).sort("Date")
    holdout = pl.read_csv(config.HOLDOUT_CSV, try_parse_dates=True).sort("Date")
    if development["Date"].max() >= holdout["Date"].min():
        raise RuntimeError("development overlaps the hold-out")
    if int(development.height) != int(summary["development_rows"]):
        raise RuntimeError("development row count does not match the saved model summary")

    model = _fit(summary, development)
    y_true = holdout["Target"].to_numpy().astype(np.int32)
    probability = model.predict_proba(holdout.select(summary["features"]).to_numpy())[:, 1]
    prediction = (probability >= 0.5).astype(np.int32)
    previous = _previous_direction(holdout)

    rules = [
        _rule("XGBoost", y_true, prediction, probability),
        _rule("Always up", y_true, np.ones_like(y_true), np.ones(len(y_true), dtype=float)),
        _rule("Repeat previous direction", y_true, previous, previous.astype(float)),
    ]
    _save_figures(y_true, probability, rules)

    gap = pl.read_csv(config.GAP_CSV, try_parse_dates=True).sort("Date")
    windows = {
        "development_rows": int(development.height),
        "development_start": _iso(development["Date"][0]),
        "development_end": _iso(development["Date"][-1]),
        "gap_rows": int(gap.height),
        "gap_start": _iso(gap["Date"][0]),
        "gap_end": _iso(gap["Date"][-1]),
        "holdout_rows": int(holdout.height),
        "holdout_start": _iso(holdout["Date"][0]),
        "holdout_end": _iso(holdout["Date"][-1]),
        "up_share": float(y_true.mean()),
    }
    payload = {
        "features": summary["features"],
        "best_params": summary["best_params"],
        "random_state": config.RANDOM_STATE,
        "threshold": 0.5,
        "development_cv_auc": summary["best_auc"],
        "windows": windows,
        "rules": [
            {key: value for key, value in rule.items() if key != "report"}
            | {"precision_up": rule["report"]["up"]["precision"], "recall_up": rule["report"]["up"]["recall"]}
            for rule in rules
        ],
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_text(rules)
    print(
        f"hold-out AUC {rules[0]['auc']:.4f}  accuracy {rules[0]['accuracy']:.4f}  "
        f"always-up {rules[1]['accuracy']:.4f}  repeat {rules[2]['accuracy']:.4f}"
    )
    print(f"wrote: {config.METRICS_JSON}")
    print(f"wrote: {config.HOLDOUT_TXT}")
    print(f"wrote: {config.ROC_FIGURE}")
    print(f"wrote: {config.CONFUSION_FIGURE}")
    _write_chapter(summary, rules, windows)


if __name__ == "__main__":
    main()
