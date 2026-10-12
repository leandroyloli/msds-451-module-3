# 5. Feature selection

Selection uses the 9 columns from the previous chapter and the 3,723 development sessions. The hold-out file is not read.

## How a subset is scored

Each subset is fit by logistic regression and scored on sessions the fit did not see. Development is cut into the same 5 time-series folds used later for XGBoost, with a gap of 10 sessions. In each fold the model is fit on earlier sessions and scored on the next block. The score is the log loss,

$$\ell = -\frac{1}{n}\sum_t \bigl[y_t \ln \hat p_t + (1 - y_t) \ln(1 - \hat p_t)\bigr],$$

averaged over the 5 folds. Lower is better. A model with no column predicts the up share of its training block every day. Its log loss, 0.69361, is the baseline.

## Forward search

The search starts with no column. At each step it adds the one column that lowers the log loss most. It stops when no column lowers it further. The gain is how far the log loss falls below the baseline, in thousandths.

| Step | Added | Mean log loss | Gain (×1000) |
| --- | --- | --- | --- |
| 1 | Ret1 | 0.69304 | 0.575 |
| 2 | Intraday1 | 0.69303 | 0.581 |

The search stops at Ret1, Intraday1, with a gain of 0.581.

## Noise gate

A search like this always finds some gain, even when the columns carry no information. The gate asks how large a gain noise alone produces.

The label column is shifted in time by at least 252 sessions, so each day's features line up with the label of another day. The forward search is then run again on the shifted label. This is repeated 200 times. The gains from these runs are what noise alone produces. Their median is 0.143, and 95 percent of them are below 1.057. A selection is kept only if its gain is above that line.

The actual gain, 0.581, is below that line. 19 percent of the shifted runs reach a gain at least as large. The search therefore does not single out any column, and the next chapter fits XGBoost on all 9 columns.

## A fair estimate

The columns above were chosen with the whole development window. Scoring them on the same window would be too generous. To get a fair estimate, the full procedure, search and gate, is repeated inside each fold using only that fold's training sessions. When the gate fails, the fold uses all 9 columns. The resulting logistic regression is scored on the next block, which the procedure did not see.

| Fold | Scored block | Forward pick | Gate passed | Columns used | ROC AUC | Gain (×1000) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2010-04-19 to 2013-03-06 | Ret1 | No | all 9 | 0.4782 | -17.347 |
| 2 | 2013-03-07 to 2015-11-04 | Ret1 | No | all 9 | 0.4869 | -0.781 |
| 3 | 2015-11-05 to 2018-11-28 | none | No | all 9 | 0.5507 | 2.636 |
| 4 | 2018-11-29 to 2021-06-04 | Ret1 | No | all 9 | 0.5127 | -8.595 |
| 5 | 2021-06-07 to 2023-12-28 | Ret1 | No | all 9 | 0.5063 | 0.620 |

The gate passes in 0 of 5 folds. The mean ROC AUC is 0.5070, and the mean gain is -4.693. A negative gain means the model did worse than predicting the training up share.
