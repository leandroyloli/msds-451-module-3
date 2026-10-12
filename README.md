# MSDS 451 Programming Assignment 1

Predict the next-day direction of WEGE3.SA (WEG ON, B3) from daily open, high, low, close, and volume. The report is `report/msds451-pa1.pdf`. The same report is in `report/msds451-pa1.html`, which can be opened in a browser.

Clone URL: https://github.com/leandroyloli/msds-451-module-3.git

## Layout

| Path | Contents |
| --- | --- |
| `pipeline/01_config.py` | Symbol, dates, split, search ranges, and file paths |
| `pipeline/02_data.py` | Download daily prices from Yahoo Finance |
| `pipeline/03_features.py` | Build the 25 candidate columns and the direction label |
| `pipeline/04_split.py` | Cut development, a 10-session gap, and the hold-out |
| `pipeline/05_universe.py` | Drift screen on the development window; writes the observable universe |
| `pipeline/06_selection.py` | Forward selection by out-of-sample log loss, noise gate, nested estimate |
| `pipeline/07_model.py` | Optuna search for XGBoost, one-standard-error choice |
| `pipeline/08_evaluate.py` | Fit the chosen model once, score the hold-out once, bootstrap intervals |
| `pipeline/09_report.py` | Render `report/msds451-pa1.html` and print `report/msds451-pa1.pdf` |
| `pipeline/10_run.py` | Run layers 03 through 09 in order; `--download` also runs 02 |
| `data/` | Prices, features, development, gap, and hold-out CSV files |
| `output/` | JSON summaries and `holdout_report.txt` |
| `figures/` | Hold-out ROC curve |
| `report/` | Chapter notes and the report in HTML and PDF |

`01_config.py` holds the premises that change the sample. `DEV_END` is the last calendar date kept in development. `GAP` is the number of following sessions left out of both the fit and the score. `UNIVERSE_MAX_OUTSIDE` is the drift limit of the universe screen. `N_SHIFTS` and `GATE_QUANTILE` set the noise gate of the selection.

## Run

Python 3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python pipeline/10_run.py
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python pipeline/10_run.py
```

`10_run.py` reuses the saved price file, so a rerun gives the same sample. Pass `--download` to fetch the prices from Yahoo Finance again; Yahoo can revise history, so the sample may then change. The CSV files, JSON summaries, figure, and HTML and PDF report already in the repository are the result of that run. The PDF is printed from the HTML by a headless Edge or Chrome; MathJax needs network access to render the equations.

Rebuild the HTML and PDF report from the saved chapters:

```bash
python pipeline/09_report.py
```

Repeat only the hold-out score, using the saved features and hyperparameters:

```bash
python pipeline/08_evaluate.py
```

## What the run checks

`03_features.py` checks the return, range, and target formulas against the raw prices before it drops incomplete rows. `04_split.py` stops if development is empty, if fewer than `GAP` sessions follow `DEV_END`, or if the hold-out is empty. `05_universe.py` stops if no candidate passes the drift screen. `06_selection.py` compares its logistic fit with scikit-learn's unpenalized fit and stops if a fitted probability differs by more than 0.001. `08_evaluate.py` stops if development overlaps the hold-out, if the development row count does not match the saved model summary, or if the first hold-out day does not follow the last gap day.

The hold-out file is not read by the drift screen, by feature selection, or by the hyperparameter search.

## AI assistants

I used AI assistants from Anthropic, Cursor, and OpenAI to help with the code and with the writing. The plan, the strategy, and the decisions were mine. I sometimes debated those choices with the models.
