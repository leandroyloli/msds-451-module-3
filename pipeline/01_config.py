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

# Observable universe. A candidate is kept when the last third of development
# stays inside the 1st to 99th percentile range of the first third.
# A stable column leaves about 2 percent outside; the limit is five times that.
UNIVERSE_QUANTILES = (0.01, 0.99)
UNIVERSE_MAX_OUTSIDE = 0.10

# Forward selection and its noise gate.
# Each shift rolls the label by at least MIN_SHIFT sessions against the features.
N_SHIFTS = 200
MIN_SHIFT = 252
GATE_QUANTILE = 0.95

# XGBoost search ranges.
XGB_MAX_DEPTH = (1, 6)
XGB_MIN_CHILD_WEIGHT = (1, 100)
XGB_SUBSAMPLE = (0.5, 1.0)
XGB_LEARNING_RATE = (0.005, 0.1)
XGB_N_ESTIMATORS = (50, 1000)

# Hold-out intervals: block bootstrap.
N_BOOT = 2000
BOOT_BLOCK = 20

PRICES_CSV = DATA_DIR / f"{FILE_STEM}_prices.csv"
FEATURES_CSV = DATA_DIR / f"{FILE_STEM}_features.csv"
DEVELOPMENT_CSV = DATA_DIR / f"{FILE_STEM}_development.csv"
GAP_CSV = DATA_DIR / f"{FILE_STEM}_gap.csv"
HOLDOUT_CSV = DATA_DIR / f"{FILE_STEM}_holdout.csv"
FEATURES_SUMMARY_JSON = OUTPUT_DIR / "features_summary.json"
SPLIT_SUMMARY_JSON = OUTPUT_DIR / "split_summary.json"
UNIVERSE_SUMMARY_JSON = OUTPUT_DIR / "universe_summary.json"
UNIVERSE_CHAPTER_MD = REPORT_DIR / "04-universe.md"
SELECTION_SUMMARY_JSON = OUTPUT_DIR / "selection_summary.json"
SELECTION_CHAPTER_MD = REPORT_DIR / "05-selection.md"
MODEL_SUMMARY_JSON = OUTPUT_DIR / "model_summary.json"
MODEL_CHAPTER_MD = REPORT_DIR / "06-model.md"
EVAL_CHAPTER_MD = REPORT_DIR / "07-holdout.md"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
HOLDOUT_TXT = OUTPUT_DIR / "holdout_report.txt"
ROC_FIGURE = FIGURES_DIR / "holdout_roc.svg"
INTERVAL_FIGURE = FIGURES_DIR / "holdout_intervals.svg"
REPORT_HTML = REPORT_DIR / "msds451-pa1.html"
REPORT_PDF = REPORT_DIR / "msds451-pa1.pdf"
PROBLEM_MD = REPORT_DIR / "01-problem.md"
DATA_MD = REPORT_DIR / "02-data.md"
FEATURES_CHAPTER_MD = REPORT_DIR / "03-features.md"
CONCLUSION_MD = REPORT_DIR / "08-conclusion.md"

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

# Ten return, volatility, and liquidity columns from the literature.
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

# All 25 candidates. 05_universe.py decides which of them enter selection.
CANDIDATE_FEATURES = LEVEL_FEATURES + LITERATURE_FEATURES
