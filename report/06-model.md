# 6. XGBoost

The classifier uses the 9 columns passed on by the previous chapter: Ret1, Ret21, Mom12_1, RVol21, Parkinson21, LogVolChg1, Range1, Intraday1, Overnight1. Optuna runs 100 trials on the 3,723 development sessions. Each trial is a different combination of tree depth, minimum child weight, row subsample, learning rate, and number of trees. The next trial is aimed near combinations that already scored well. The score is the mean ROC AUC across 5 time-series folds, with a gap of 10 sessions:

$$\overline{\mathrm{AUC}} = \frac{1}{5}\sum_{k=1}^{5}\mathrm{AUC}_k.$$

Tree depth runs from 1 to 6. The subsample is kept between 0.5 and 1. The seed is 21. The hold-out file is not read.

Many trials score almost the same, so the single best one is partly luck. The search keeps the simplest trial whose score is within one standard error of the best: shallower trees first, then fewer trees. The best mean AUC is 0.5115, and 83 of the 100 trials fall within one standard error of it.

The chosen settings are depth 1, minimum child weight 3, subsample 0.782, learning rate 0.007, and 51 trees. Their mean AUC is 0.5077. A logistic regression on the same columns and folds reaches 0.5070.

## Ten best trials

| Rank | Mean AUC | Depth | Min child | Subsample | Learning rate | Trees |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5115 | 1 | 1 | 0.677 | 0.006 | 193 |
| 2 | 0.5108 | 1 | 1 | 0.697 | 0.005 | 182 |
| 3 | 0.5107 | 1 | 3 | 0.678 | 0.007 | 171 |
| 4 | 0.5106 | 1 | 3 | 0.705 | 0.008 | 169 |
| 5 | 0.5105 | 1 | 1 | 0.724 | 0.006 | 76 |
| 6 | 0.5105 | 1 | 1 | 0.681 | 0.006 | 161 |
| 7 | 0.5105 | 1 | 2 | 0.729 | 0.006 | 187 |
| 8 | 0.5105 | 1 | 3 | 0.697 | 0.005 | 197 |
| 9 | 0.5103 | 1 | 2 | 0.673 | 0.005 | 53 |
| 10 | 0.5103 | 1 | 1 | 0.924 | 0.007 | 399 |
