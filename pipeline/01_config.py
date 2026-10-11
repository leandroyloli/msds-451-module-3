"""Settings shared by every later layer."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "output"
REPORT_DIR = ROOT / "report"

SYMBOL = "WEGE3.SA"
FILE_STEM = "wege3_sa"

START = "2000-01-01"
END = "2026-10-11"

# Split premises. Change these and rerun pipeline/04_split.py.
# DEV_END is the last calendar date kept in development.
# GAP is the number of later sessions left out of both windows.
DEV_END = "2023-12-31"
GAP = 10

N_SPLITS = 5
N_TRIALS = 100
RANDOM_STATE = 21

# XGBoost search ranges. subsample stays in [0.5, 1].
XGB_MAX_DEPTH = (3, 9)
XGB_MIN_CHILD_WEIGHT = (1, 9)
XGB_SUBSAMPLE = (0.5, 1.0)
XGB_LEARNING_RATE = (0.01, 0.11)
XGB_N_ESTIMATORS = (100, 999)

PRICES_CSV = DATA_DIR / f"{FILE_STEM}_prices.csv"
FEATURES_CSV = DATA_DIR / f"{FILE_STEM}_features.csv"
DEVELOPMENT_CSV = DATA_DIR / f"{FILE_STEM}_development.csv"
GAP_CSV = DATA_DIR / f"{FILE_STEM}_gap.csv"
HOLDOUT_CSV = DATA_DIR / f"{FILE_STEM}_holdout.csv"
FEATURES_SUMMARY_JSON = OUTPUT_DIR / "features_summary.json"
SPLIT_SUMMARY_JSON = OUTPUT_DIR / "split_summary.json"
SELECTION_SUMMARY_JSON = OUTPUT_DIR / "selection_summary.json"
SELECTION_CHAPTER_MD = REPORT_DIR / "04-selection.md"
SELECTION_CHAPTER_PDF = REPORT_DIR / "04-selection.pdf"
MODEL_SUMMARY_JSON = OUTPUT_DIR / "model_summary.json"
MODEL_CHAPTER_MD = REPORT_DIR / "05-model.md"
MODEL_CHAPTER_PDF = REPORT_DIR / "05-model.pdf"
EVAL_CHAPTER_MD = REPORT_DIR / "06-holdout.md"
EVAL_CHAPTER_PDF = REPORT_DIR / "06-holdout.pdf"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
HOLDOUT_TXT = OUTPUT_DIR / "holdout_report.txt"
ROC_FIGURE = FIGURES_DIR / "holdout_roc.png"
CONFUSION_FIGURE = FIGURES_DIR / "holdout_confusion.png"
REPORT_PDF = REPORT_DIR / "msds451-pa1.pdf"
PROBLEM_MD = REPORT_DIR / "01-problem.md"
DATA_MD = REPORT_DIR / "02-data.md"
FEATURES_CHAPTER_MD = REPORT_DIR / "03-features.md"
FEATURES_CHAPTER_PDF = REPORT_DIR / "03-features.pdf"
CONCLUSION_MD = REPORT_DIR / "07-conclusion.md"

# Fifteen price-level columns from the course example.
LEVEL_FEATURES = [
    "CloseLag1",
    "CloseLag2",
    "CloseLag3",
    "HMLLag1",
    "HMLLag2",
    "HMLLag3",
    "OMCLag1",
    "OMCLag2",
    "OMCLag3",
    "VolumeLag1",
    "VolumeLag2",
    "VolumeLag3",
    "CloseEMA2",
    "CloseEMA4",
    "CloseEMA8",
]

# Ten return, volatility, and liquidity columns. Searched together with LEVEL_FEATURES.
LITERATURE_FEATURES = [
    "Ret1",
    "Ret21",
    "Mom12_1",
    "RVol21",
    "Parkinson21",
    "LogVolChg1",
    "Amihud21",
    "Range1",
    "Intraday1",
    "Overnight1",
]

MODEL_FEATURES = LEVEL_FEATURES + LITERATURE_FEATURES

# These six track the same price path. A candidate subset may include at most one.
PRICE_LEVEL_ONE_OF = [
    "CloseLag1",
    "CloseLag2",
    "CloseLag3",
    "CloseEMA2",
    "CloseEMA4",
    "CloseEMA8",
]
