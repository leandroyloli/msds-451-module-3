# 8. Conclusion

The drift screen keeps 9 of the 25 candidates. The 16 it removes are measured in reais, in shares, or per real traded. They follow the growth of the company, not the next session.

On those 9 columns, no subset beats noise. The forward search picks Ret1 and Intraday1, but 19 percent of the runs on shifted labels gain at least as much. XGBoost therefore uses all 9 columns. Its mean ROC AUC on the development folds is 0.5077, and the fair estimate from the nested folds is 0.5070.

On the 685 hold-out sessions XGBoost has ROC AUC 0.5303 and accuracy 0.5080, the same accuracy as always up. Its 95 percent range for ROC AUC, 0.4921 to 0.5729, includes 0.5. The logistic regression has ROC AUC 0.5267 and accuracy 0.5226, and its range also includes 0.5. Neither classifier shows skill. In this design, the next-session direction of WEGE3.SA is not predictable from daily open, high, low, close, and volume.

The course example reports classification accuracy of 81.8 percent and an AUC of 0.82 after fitting and scoring the same rows. Its time-series cross-validation accuracy was about 50 to 51 percent, and the example states that training-set predictions are better than should be expected on new data. The hold-out numbers in this report sit with that cross-validation result.

## References

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.

Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.

Jegadeesh, N., and Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.

Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61–65.

Campbell, J. Y., Grossman, S. J., and Wang, J. (1993). Trading volume and serial correlation in stock returns. *Quarterly Journal of Economics*, 108(4), 905–939.

Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.

## Software

Prices were downloaded with yfinance. Candidate columns were built with Polars. The forward search checks its logistic fit against scikit-learn's unpenalized logistic regression. The classifier is XGBoost, and the hyperparameter search is Optuna. The figures use Matplotlib.
