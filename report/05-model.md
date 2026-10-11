# 5. XGBoost

The classifier uses the three columns chosen in the previous chapter: VolumeLag1, Ret1, CloseEMA8. Optuna runs 100 trials on the 3,723 development sessions. Each trial is a different combination of tree depth, minimum child weight, row subsample, learning rate, and number of trees. The next trial is aimed near combinations that already scored well. The score is the mean ROC AUC across 5 time-series folds, with a gap of 10 sessions. The subsample is kept between 0.5 and 1. The seed is 21. The hold-out file is not read.

The best mean AUC on the development folds is 0.5335, with a spread across folds of 0.0307. The chosen settings are depth 9, minimum child weight 9, subsample 0.595, learning rate 0.013, and 925 trees.

## Ten best trials

| Rank | Mean AUC | Depth | Min child | Subsample | Learning rate | Trees |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5335 | 9 | 9 | 0.595 | 0.013 | 925 |
| 2 | 0.5334 | 8 | 9 | 0.576 | 0.017 | 881 |
| 3 | 0.5323 | 9 | 9 | 0.573 | 0.034 | 676 |
| 4 | 0.5321 | 8 | 9 | 0.555 | 0.023 | 845 |
| 5 | 0.5318 | 9 | 9 | 0.513 | 0.018 | 889 |
| 6 | 0.5318 | 9 | 9 | 0.592 | 0.012 | 991 |
| 7 | 0.5317 | 9 | 9 | 0.548 | 0.034 | 753 |
| 8 | 0.5315 | 9 | 9 | 0.573 | 0.014 | 996 |
| 9 | 0.5312 | 8 | 8 | 0.524 | 0.016 | 847 |
| 10 | 0.5310 | 9 | 8 | 0.555 | 0.025 | 791 |
