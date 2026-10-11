# 7. Conclusion

The selected logistic model is a small improvement on an intercept. Its AIC is 5157.46. An intercept-only model on the same development sessions has AIC 5162.88. The XGBoost search, scored by mean ROC AUC on five time-series folds of the development window, reaches 0.5335.

On the 685 hold-out sessions the fitted classifier has accuracy 0.4847 and ROC AUC 0.4752. Always predicting an up day has accuracy 0.5080. Repeating the previous session's direction has accuracy 0.4599. On sessions the model did not see, the constant up forecast is ahead of the fitted classifier.

The course example reports classification accuracy of 81.8 percent and an AUC of 0.82 after fitting and scoring the same rows. Its time-series cross-validation accuracy was about 50 to 51 percent, and the example states that training-set predictions are better than should be expected on new data. The hold-out numbers in this report sit with that cross-validation result.

## References

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. Journal of Finance, 45(3), 881-898.

Lehmann, B. N. (1990). Fads, martingales, and market efficiency. Quarterly Journal of Economics, 105(1), 1-28.

Jegadeesh, N., and Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. Journal of Finance, 48(1), 65-91.

Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. Journal of Business, 53(1), 61-65.

Campbell, J. Y., Grossman, S. J., and Wang, J. (1993). Trading volume and serial correlation in stock returns. Quarterly Journal of Economics, 108(4), 905-939.

Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. Journal of Financial Markets, 5(1), 31-56.

Prices were downloaded with yfinance. Candidate columns were built with Polars. The AIC search checks its fit against scikit-learn's unpenalized logistic regression. The classifier is XGBoost, and the hyperparameter search is Optuna. The figures use Matplotlib.
