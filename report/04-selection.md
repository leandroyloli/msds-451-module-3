# 4. Feature selection

The search uses the 3,723 development sessions only. Each candidate subset is fit by an unpenalized logistic regression, which is the maximum likelihood the AIC formula assumes. The subset with the smallest AIC is kept. Columns are standardized with the mean and standard deviation of those development sessions. For an unpenalized logistic regression with an intercept, that rescaling does not change the maximized likelihood. The hold-out file is not read.

## Combinations that repeat the same price

Six columns track one price path: CloseLag1, CloseLag2, CloseLag3, CloseEMA2, CloseEMA4, and CloseEMA8. On the development window the smallest absolute correlation among them is 0.999. Adding the other five to CloseLag1 raises the log-likelihood by 1.45. AIC needs a gain above 1 for each added column, so a second copy does not pay for itself. CloseLag1 with CloseLag3 gains 0.26. CloseLag1 with CloseEMA8 gains 0.00.

The search allows at most one of those six. The other 19 columns stay free, including next to that one price column. High minus low together with the close still belongs in the search: that pair gains 1.15 over high minus low alone, which clears the hurdle of 1.

Subsets of all 25 columns number 33,554,431. With at most one price-level column the number is 3,670,015. A second price-level column cannot improve the AIC, so the cut does not change the winner.

## Selected subset

The smallest AIC is 5157.46, on VolumeLag1, Ret1, CloseEMA8. An intercept-only model on the same 3,723 sessions has AIC 5162.88. The winner improves on that null model by 5.42.

| Rank | AIC | Features |
| --- | --- | --- |
| 1 | 5157.46 | VolumeLag1, Ret1, CloseEMA8 |
| 2 | 5157.58 | VolumeLag1, Ret1, CloseEMA4 |
| 3 | 5157.61 | VolumeLag1, Ret1, CloseLag3 |
| 4 | 5157.64 | VolumeLag1, Ret1, CloseEMA2 |
| 5 | 5157.66 | VolumeLag1, Ret1, CloseLag2 |
| 6 | 5157.66 | VolumeLag1, Ret1, CloseLag1 |
| 7 | 5158.00 | VolumeLag1, VolumeLag3, Ret1, CloseEMA8 |
| 8 | 5158.13 | VolumeLag1, VolumeLag3, Ret1, CloseEMA4 |
| 9 | 5158.17 | VolumeLag1, VolumeLag3, Ret1, CloseLag3 |
| 10 | 5158.19 | VolumeLag1, VolumeLag3, Ret1, CloseEMA2 |

The search scored 3,670,015 subsets in 35.3 minutes. 47,799 fits did not settle and were discarded.
