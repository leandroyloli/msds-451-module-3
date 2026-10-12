"""Fit the chosen model once and score the hold-out once.

Reads the columns from 06_selection.py and the settings from 07_model.py.
Does not rerun either search. Intervals come from a block bootstrap of the
hold-out sessions. The criterion for skill is fixed in advance: the lower end
of the 95 percent interval for ROC AUC must lie above 0.5.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_CONFIG_PATH = Path(__file__).with_name("01_config.py")
_spec = importlib.util.spec_from_file_location("config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

_EPS = 1e-12


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


def _fit_xgb(summary: dict, x: np.ndarray, y: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        **summary["best_params"],
    )
    return model.fit(x, y)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        return 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))


def _loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _blocks(n: int, rng: np.random.Generator) -> np.ndarray:
    block = config.BOOT_BLOCK
    starts = rng.integers(0, n - block + 1, size=n // block + 1)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def _bootstrap(y_true: np.ndarray, scored: list[dict]) -> dict:
    """Block bootstrap: 95 percent intervals for each rule and for the gap to always up."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    draws: dict[str, list[float]] = {}
    for _ in range(config.N_BOOT):
        idx = _blocks(len(y_true), rng)
        y = y_true[idx]
        always = float(y.mean())
        for rule in scored:
            name = rule["name"]
            acc = float(np.mean(rule["prediction"][idx] == y))
            draws.setdefault(f"{name}|accuracy", []).append(acc)
            draws.setdefault(f"{name}|accuracy_gap", []).append(acc - always)
            draws.setdefault(f"{name}|auc", []).append(_auc(y, rule["score"][idx]))
            if rule["probability"] is not None:
                draws.setdefault(f"{name}|log_loss", []).append(_loss(y, rule["probability"][idx]))
    return {key: [float(v) for v in np.percentile(values, [2.5, 97.5])] for key, values in draws.items()}


def _rule(name: str, y_true: np.ndarray, prediction: np.ndarray, score: np.ndarray, probability) -> dict:
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    down_hit, down_miss, up_miss, up_hit = (int(value) for value in matrix.ravel())
    report = classification_report(
        y_true,
        prediction,
        labels=[0, 1],
        target_names=["down or flat", "up"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "auc": _auc(y_true, score),
        "log_loss": None if probability is None else _loss(y_true, probability),
        "down_hit": down_hit,
        "down_miss": down_miss,
        "up_miss": up_miss,
        "up_hit": up_hit,
        "up_call_share": float(np.mean(prediction)),
        "report": report,
        "matrix": matrix.tolist(),
        "prediction": prediction,
        "score": score,
        "probability": probability,
    }


def _save_figure(y_true: np.ndarray, rules: list[dict]) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    figure, axis = plt.subplots(figsize=(5.6, 5.6), facecolor="white")
    axis.set_facecolor("white")
    axis.plot([0, 1], [0, 1], color="#c5d2c9", linestyle=(0, (4, 4)), linewidth=1.1, label="Chance", zorder=1)
    curves = (("#14231c", "-"), ("#7d8c84", "--"))
    for rule, (color, style) in zip(rules[:2], curves):
        fpr, tpr, _ = roc_curve(y_true, rule["score"])
        axis.plot(fpr, tpr, color=color, linestyle=style, linewidth=1.8, label=f"{rule['name']}  {rule['auc']:.3f}", zorder=2)
    marks = (("#1f6b56", "o"), ("#8f4038", "s"))
    for rule, (color, marker) in zip(rules[2:], marks):
        actual_down = rule["down_hit"] + rule["down_miss"]
        actual_up = rule["up_hit"] + rule["up_miss"]
        axis.scatter(
            rule["down_miss"] / actual_down,
            rule["up_hit"] / actual_up,
            s=42,
            color=color,
            marker=marker,
            zorder=3,
            clip_on=False,
            label=rule["name"],
        )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_aspect("equal")
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlabel("False positive rate", color="#14231c", fontsize=10)
    axis.set_ylabel("True positive rate", color="#14231c", fontsize=10)
    axis.tick_params(colors="#5c6b63", labelsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#d5ded8")
    axis.grid(True, color="#e7eeea", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    figure.tight_layout()
    figure.savefig(config.ROC_FIGURE, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_interval_figure(rules: list[dict], intervals: dict) -> None:
    """ROC AUC of each rule with its 95 percent block-bootstrap interval."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    shown = [rule for rule in rules if len(np.unique(rule["score"])) > 1]
    figure, axis = plt.subplots(figsize=(7.2, 0.62 * len(shown) + 1.2), facecolor="white")
    axis.set_facecolor("white")
    colors = ("#14231c", "#7d8c84", "#8f4038", "#1f6b56")
    for row, rule in enumerate(reversed(shown)):
        lo, hi = intervals[f"{rule['name']}|auc"]
        color = colors[rules.index(rule)]
        axis.plot([lo, hi], [row, row], color=color, linewidth=2.2, solid_capstyle="round", zorder=2)
        axis.scatter([rule["auc"]], [row], s=46, color=color, zorder=3)
        axis.text(hi + 0.004, row, f"{rule['auc']:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=8.5, color="#5c6b63")
    axis.axvline(0.5, color="#c5d2c9", linestyle=(0, (4, 4)), linewidth=1.1, zorder=1)
    axis.set_yticks(range(len(shown)))
    axis.set_yticklabels([rule["name"] for rule in reversed(shown)], fontsize=9, color="#14231c")
    low = min(intervals[f"{rule['name']}|auc"][0] for rule in shown)
    high = max(intervals[f"{rule['name']}|auc"][1] for rule in shown)
    axis.set_xlim(min(low, 0.5) - 0.01, max(high, 0.5) + 0.07)
    axis.set_ylim(-0.6, len(shown) - 0.4)
    axis.set_xlabel("Hold-out ROC AUC", color="#14231c", fontsize=10)
    axis.tick_params(axis="x", colors="#5c6b63", labelsize=9)
    axis.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#d5ded8")
    axis.grid(True, axis="x", color="#e7eeea", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(config.INTERVAL_FIGURE, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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
    total = rule["down_hit"] + rule["down_miss"] + rule["up_miss"] + rule["up_hit"]
    rows.append(f"{'accuracy':16} {'':>10} {'':>10} {report['accuracy']:10.4f} {total:10d}")
    for label in ("macro avg", "weighted avg"):
        item = report[label]
        rows.append(
            f"{label:16} {item['precision']:10.4f} {item['recall']:10.4f} "
            f"{item['f1-score']:10.4f} {int(item['support']):10d}"
        )
    return "\n".join(rows)


def _write_text(rules: list[dict], intervals: dict) -> None:
    lines = []
    for rule in rules:
        name = rule["name"]
        lines.append(name)
        lines.append("Confusion Matrix")
        lines.append(str(np.array(rule["matrix"])))
        lines.append(_report_text(rule))
        lo, hi = intervals[f"{name}|auc"]
        lines.append(f"ROC AUC {rule['auc']:.4f}  95% interval {lo:.4f} to {hi:.4f}")
        if rule["log_loss"] is not None:
            lo, hi = intervals[f"{name}|log_loss"]
            lines.append(f"log loss {rule['log_loss']:.5f}  95% interval {lo:.5f} to {hi:.5f}")
        lines.append("")
    config.HOLDOUT_TXT.write_text("\n".join(lines), encoding="utf-8")


def _interval(pair: list[float], digits: int = 4) -> str:
    return f"{pair[0]:.{digits}f} to {pair[1]:.{digits}f}"


def _write_chapter(summary: dict, rules: list[dict], intervals: dict, windows: dict, skill: bool) -> None:
    model, logit, always, repeat = rules
    counts = "\n".join(
        f"| {rule['name']} | {rule['accuracy']:.4f} | {rule['auc']:.4f} | "
        f"{rule['down_hit']} | {rule['down_miss']} | {rule['up_miss']} | {rule['up_hit']} |"
        for rule in rules
    )
    ranges = "\n".join(
        f"| {rule['name']} | {rule['auc']:.4f} | {_interval(intervals[rule['name'] + '|auc'])} | "
        + (f"{rule['log_loss']:.4f} |" if rule["log_loss"] is not None else "n/a |")
        for rule in (model, logit, always, repeat)
    )
    best = summary["best_params"]
    lo, hi = intervals[f"{model['name']}|auc"]
    width = (hi - lo) / 2
    calls = (
        "All its fitted probabilities are above one half, so it calls every session up, like always up."
        if model["up_call_share"] == 1.0
        else f"It calls an up day on {model['up_call_share']:.4f} of the sessions."
    )
    verdict = (
        f"The XGBoost range runs from {lo:.4f} to {hi:.4f}. It sits above 0.5, so the classifier shows skill on the hold-out."
        if skill
        else f"The XGBoost range runs from {lo:.4f} to {hi:.4f}. It includes 0.5, so the classifier shows no skill on the hold-out."
    )
    text = f"""# 7. Hold-out

The classifier is fit once on the {windows["development_rows"]:,} development sessions, from {windows["development_start"]} through {windows["development_end"]}. The settings are the ones chosen in the previous chapter: depth {best["max_depth"]}, minimum child weight {best["min_child_weight"]}, subsample {best["subsample"]:.3f}, learning rate {best["learning_rate"]:.3f}, and {best["n_estimators"]} trees. The columns are {", ".join(summary["features"])}. The seed is {summary["random_state"]}. The fit does not see the hold-out.

The score uses the {windows["holdout_rows"]:,} hold-out sessions, from {windows["holdout_start"]} through {windows["holdout_end"]}. The {windows["gap_rows"]} sessions from {windows["gap_start"]} through {windows["gap_end"]} stay out of the fit and out of the score. A session is called up when the fitted probability satisfies

$$\\hat y_t = \\mathbf{{1}}\\{{\\hat p_t \\ge 1/2\\}}.$$

Three reference rules are scored with it. The logistic regression uses the same columns and has no tuned setting. Always up predicts an up day every session. Repeat previous direction predicts that today's direction equals the direction of the immediately previous session. For the first hold-out day, that previous session is {windows["gap_end"]}.

Down hit counts down-or-flat days predicted as down-or-flat. Down miss counts those days predicted as up. Up miss counts up days predicted as down-or-flat. Up hit counts up days predicted as up. With up as the positive class,

$$\\mathrm{{TPR}} = \\frac{{\\text{{up hit}}}}{{\\text{{up hit}} + \\text{{up miss}}}}, \\qquad \\mathrm{{FPR}} = \\frac{{\\text{{down miss}}}}{{\\text{{down miss}} + \\text{{down hit}}}}.$$

The always-up rule assigns the same score to every session, so its ROC AUC is 0.5.

## Hold-out scores

| Rule | Accuracy | ROC AUC | Down hit | Down miss | Up miss | Up hit |
| --- | --- | --- | --- | --- | --- | --- |
{counts}

The hold-out share of up days is {windows["up_share"]:.4f}. Always up is therefore right on {always["accuracy"]:.4f} of the hold-out sessions. XGBoost is right on {model["accuracy"]:.4f}, with ROC AUC {model["auc"]:.4f}. {calls} The logistic regression is right on {logit["accuracy"]:.4f}, with ROC AUC {logit["auc"]:.4f}. Repeat previous direction is right on {repeat["accuracy"]:.4f}.

## Chance or skill

With {windows["holdout_rows"]} sessions, part of any difference is chance. The hold-out is resampled {config.N_BOOT:,} times in blocks of {config.BOOT_BLOCK} consecutive sessions, and each rule is scored again on every sample. The middle 95 percent of those scores is the range below. The classifier counts as skilled only if its whole ROC AUC range sits above 0.5. That rule was set before the hold-out was scored. Log loss measures the fitted probabilities directly; lower is better.

| Rule | ROC AUC | 95% range | Log loss |
| --- | --- | --- | --- |
{ranges}

![Figure 1. Hold-out ROC AUC with its 95 percent range. The dashed line is 0.5.](../figures/holdout_intervals.svg)

{verdict} With {windows["holdout_rows"]} sessions, a range clears 0.5 only when the ROC AUC is near {0.5 + width:.2f} or higher.

![Figure 2. Hold-out ROC. The curves stay close to the diagonal.](../figures/holdout_roc.svg)
"""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_CHAPTER_MD.write_text(text, encoding="utf-8")
    print(f"wrote: {config.EVAL_CHAPTER_MD}")


def main() -> None:
    summary = _load_model_summary()
    development = pl.read_csv(config.DEVELOPMENT_CSV, try_parse_dates=True).sort("Date")
    holdout = pl.read_csv(config.HOLDOUT_CSV, try_parse_dates=True).sort("Date")
    if development["Date"].max() >= holdout["Date"].min():
        raise RuntimeError("development overlaps the hold-out")
    if int(development.height) != int(summary["development_rows"]):
        raise RuntimeError("development row count does not match the saved model summary")

    features = summary["features"]
    x_dev = development.select(features).to_numpy()
    y_dev = development["Target"].to_numpy().astype(np.int32)
    x_hold = holdout.select(features).to_numpy()
    y_true = holdout["Target"].to_numpy().astype(np.int32)

    p_xgb = _fit_xgb(summary, x_dev, y_dev).predict_proba(x_hold)[:, 1]
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=1000)).fit(x_dev, y_dev)
    p_logit = logit.predict_proba(x_hold)[:, 1]
    base = float(y_dev.mean())
    p_always = np.full(len(y_true), base)
    previous = _previous_direction(holdout)

    rules = [
        _rule("XGBoost", y_true, (p_xgb >= 0.5).astype(np.int32), p_xgb, p_xgb),
        _rule("Logistic regression", y_true, (p_logit >= 0.5).astype(np.int32), p_logit, p_logit),
        _rule("Always up", y_true, np.ones_like(y_true), p_always, p_always),
        _rule("Repeat previous direction", y_true, previous, previous.astype(float), None),
    ]
    intervals = _bootstrap(y_true, rules)
    skill = intervals["XGBoost|auc"][0] > 0.5
    _save_figure(y_true, rules)
    _save_interval_figure(rules, intervals)

    gap = pl.read_csv(config.GAP_CSV, try_parse_dates=True).sort("Date")
    windows = {
        "development_rows": int(development.height),
        "development_start": _iso(development["Date"][0]),
        "development_end": _iso(development["Date"][-1]),
        "development_up_share": base,
        "gap_rows": int(gap.height),
        "gap_start": _iso(gap["Date"][0]),
        "gap_end": _iso(gap["Date"][-1]),
        "holdout_rows": int(holdout.height),
        "holdout_start": _iso(holdout["Date"][0]),
        "holdout_end": _iso(holdout["Date"][-1]),
        "up_share": float(y_true.mean()),
    }
    hidden = {"report", "prediction", "score", "probability"}
    payload = {
        "features": features,
        "best_params": summary["best_params"],
        "random_state": config.RANDOM_STATE,
        "threshold": 0.5,
        "development_cv_auc": summary["best_auc"],
        "windows": windows,
        "bootstrap": {"draws": config.N_BOOT, "block": config.BOOT_BLOCK},
        "skill_criterion": "lower end of the 95 percent ROC AUC interval above 0.5",
        "skill": bool(skill),
        "intervals": intervals,
        "rules": [
            {key: value for key, value in rule.items() if key not in hidden}
            | {"precision_up": rule["report"]["up"]["precision"], "recall_up": rule["report"]["up"]["recall"]}
            for rule in rules
        ],
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_text(rules, intervals)
    for rule in rules:
        lo, hi = intervals[f"{rule['name']}|auc"]
        print(f"{rule['name']:26s} auc {rule['auc']:.4f} [{lo:.4f}, {hi:.4f}]  accuracy {rule['accuracy']:.4f}")
    print(f"skill: {skill}")
    print(f"wrote: {config.METRICS_JSON}")
    print(f"wrote: {config.HOLDOUT_TXT}")
    print(f"wrote: {config.ROC_FIGURE}")
    print(f"wrote: {config.INTERVAL_FIGURE}")
    _write_chapter(summary, rules, intervals, windows, skill)


if __name__ == "__main__":
    main()
