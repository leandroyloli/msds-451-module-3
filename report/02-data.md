# 2. Data and research design

Daily open, high, low, close, and volume come from Yahoo Finance for WEGE3.SA. Yahoo applies the split and dividend adjustment. The request covers 2000-01-01 through 2026-10-11. Yahoo treats the end date as exclusive, so the last session in the file is 2026-10-09. The download has 6,721 sessions.

The program is the numbered scripts in pipeline/. Shared settings are in 01_config.py. The later scripts download prices, build 25 candidate columns, cut the sample, screen the candidates for drift, select columns by an out-of-sample forward search with a noise gate, tune XGBoost, score the hold-out once, and render this report. Python cannot import a module whose name starts with a digit, so 10_run.py loads each file by name.

A row enters the feature table only when all 25 candidates are defined. That table runs from 2007-07-03 through 2026-10-09 and has 4,418 sessions. It is then cut with two settings in 01_config.py.

DEV_END is 2023-12-31. Development is every feature row on or before that date: 3,723 sessions, from 2007-07-03 through 2023-12-28. The next 10 sessions, 2024-01-02 through 2024-01-15, are a gap. They are neither fit nor scored. The remaining 685 sessions, 2024-01-16 through 2026-10-09, are the hold-out.

The drift screen, feature selection, and the hyperparameter search read the development file only. After those choices are fixed, XGBoost is fit on all 3,723 development sessions and applied once to the hold-out. No layer before the hold-out score reads the hold-out file.
