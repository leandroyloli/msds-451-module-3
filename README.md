# MSDS 451 Programming Assignment 1

Predict the next-day direction of WEGE3.SA (WEG ON, B3) from daily open, high, low, close, and volume. The report is `report/msds451-pa1.pdf`.

Clone URL: https://github.com/leandroyloli/msds-451-module-3.git

## Layout

| Path | Contents |
| --- | --- |
| `pipeline/01_config.py` | Symbol, dates, split, search ranges, and file paths |
| `pipeline/02_data.py` | Download daily prices from Yahoo Finance |
| `pipeline/03_features.py` | Build the 25 candidate columns and the direction label |
| `pipeline/04_split.py` | Cut development, a 10-session gap, and the hold-out |
| `pipeline/05_selection.py` | All-subsets AIC on the development window |
| `pipeline/06_model.py` | Optuna search for XGBoost on the development window |
| `pipeline/07_evaluate.py` | Fit the chosen model once and score the hold-out once |
| `pipeline/08_report.py` | Render `report/msds451-pa1.pdf` |
| `pipeline/09_run.py` | Run layers 02 through 08 in order |
| `data/` | Prices, features, development, gap, and hold-out CSV files |
| `output/` | JSON summaries and `holdout_report.txt` |
| `figures/` | Hold-out ROC curve and confusion matrix |
| `report/` | Chapter notes and the PDF report |

`01_config.py` holds the premises that change the sample. `DEV_END` is the last calendar date kept in development. `GAP` is the number of following sessions left out of both the fit and the score.

## Run

Python 3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python pipeline/09_run.py
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python pipeline/09_run.py
```

`09_run.py` downloads the prices again, repeats the AIC search, and repeats the Optuna search. On the machine used for this submission those two searches took about 35 minutes and about 19 minutes. The CSV files, JSON summaries, figures, and PDF already in the repository are the result of that run.

Rebuild the PDF from the saved chapters:

```bash
python pipeline/08_report.py
```

Repeat only the hold-out score, using the saved features and hyperparameters:

```bash
python pipeline/07_evaluate.py
```

## What the run checks

`03_features.py` checks the return, range, and target formulas against the raw prices before it drops incomplete rows. `04_split.py` stops if development is empty, if fewer than `GAP` sessions follow `DEV_END`, or if the hold-out is empty. `05_selection.py` compares its logistic fit with scikit-learn on a sample of subsets and stops if the AIC gap exceeds 0.05. `07_evaluate.py` stops if development overlaps the hold-out, if the development row count does not match the saved model summary, or if the first hold-out day does not follow the last gap day.

The hold-out file is not read by feature selection or by the hyperparameter search.

## AI assistants

I used Cursor, with an AI coding assistant, to implement the pipeline, run the searches, and draft the report chapters. I chose the asset (WEGE3.SA), the 25 candidate columns, the single joint AIC search, the rule of at most one price-level column, the development date, the 10-session gap, and the XGBoost search (Optuna, 100 trials, ROC AUC, seed 21). I reviewed each stage before the next one was run. The hold-out was scored once, after those choices were fixed, and was not used to change them.
