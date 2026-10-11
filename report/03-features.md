# 3. Features

The target is the direction of the close-to-close log return on WEGE3.SA. On day t the label is 1 when log(C(t) / C(t-1)) is positive, and 0 when the close is flat or down. Each of the 25 candidates is known at the close of day t-1. The open, high, low, close, and volume of day t do not enter any candidate. Values are stored at full precision.

The 25 columns are one pool. A later chapter searches their subsets jointly. The first 15 follow the course example and are measured in reais or in shares. The other 10 restate return, volatility, and liquidity in units that do not grow with the price level. The adjusted close of WEG rises from about 0.09 reais in 2000 to about 55 reais in 2026, so a lag of the price mostly tracks that rise.

## Definitions

| Feature | Definition, using information through day t-1 | Role |
| --- | --- | --- |
| CloseLag1 | Close on t-1 | Example. Yesterday's price level. |
| CloseLag2 | Close on t-2 | Example. Price level two sessions back. |
| CloseLag3 | Close on t-3 | Example. Price level three sessions back. |
| HMLLag1 | High minus low on t-1 | Example. Yesterday's range, in reais. |
| HMLLag2 | High minus low on t-2 | Example. |
| HMLLag3 | High minus low on t-3 | Example. |
| OMCLag1 | Open minus close on t-1 | Example. Yesterday's open-to-close change, in reais. |
| OMCLag2 | Open minus close on t-2 | Example. |
| OMCLag3 | Open minus close on t-3 | Example. |
| VolumeLag1 | Share volume on t-1 | Example. Undefined when the recorded volume is zero. |
| VolumeLag2 | Share volume on t-2 | Example. Same zero-volume rule. |
| VolumeLag3 | Share volume on t-3 | Example. Same zero-volume rule. |
| CloseEMA2 | Exponential moving average of the close through t-1, half-life of 1 session | Example. The jump-start names are EMA2, EMA4, and EMA8; the half-lives are 1, 2, and 4 sessions. |
| CloseEMA4 | Same average, half-life of 2 sessions | Example. |
| CloseEMA8 | Same average, half-life of 4 sessions | Example. |
| Ret1 | log(C(t-1) / C(t-2)) | One-day reversal (Jegadeesh, 1990; Lehmann, 1990). |
| Ret21 | log(C(t-1) / C(t-22)) | About one month of past return. |
| Mom12_1 | log(C(t-21) / C(t-252)) | Twelve-month return, skipping the most recent month (Jegadeesh and Titman, 1993). |
| RVol21 | Sample standard deviation of the 21 daily log returns ending on t-1 | Recent volatility. |
| Parkinson21 | Mean, over those 21 sessions, of [log(H/L)]^2 / (4 ln 2) | Range-based variance (Parkinson, 1980). It uses the high and the low. |
| LogVolChg1 | log(V(t-1) / V(t-2)), on positive volume | Volume shock, linked to short-horizon reversal (Campbell, Grossman, and Wang, 1993). |
| Amihud21 | Mean, over the 21 sessions ending on t-1, of absolute return / (close times share volume) | Illiquidity (Amihud, 2002). Each session in the window must have positive volume. |
| Range1 | (H(t-1) - L(t-1)) / C(t-1) | Yesterday's range as a fraction of price. Scale-free HML. |
| Intraday1 | log(C(t-1) / O(t-1)) | Yesterday's open-to-close return. Scale-free OMC. |
| Overnight1 | log(O(t-1) / C(t-2)) | Yesterday's overnight gap. |

## Volume and the estimation sample

Yahoo records volume of zero on 1,814 of the 6,721 downloaded sessions, including nearly every session from 2000 through 2006. A zero is treated as missing volume, not as a session with no shares traded. Volume lags, the log volume change, and the Amihud ratio are undefined on a zero print. A row is kept only when all 25 candidates are defined. Closes from before the first kept row still enter the moving averages and Mom12_1. They are history, not rows of their own. No missing value is filled in.

The kept table has 4,418 sessions, from 3 July 2007 through 9 October 2026. Of these, 2,229 (50.5 percent) are up days and 2,189 are flat or down. Within that span, 368 downloaded sessions are absent because a zero-volume print leaves at least one candidate undefined.

The same file stores the day's raw prices, the same-day range, the same-day open-minus-close, and the log return, so the candidates can be checked. Those columns are not part of the 25.

RSI, MACD, Bollinger bands, fractional differentiation, and any series that is not in this file of daily open, high, low, close, and volume are outside the candidate list. Chapter 2 cuts this table into a development window and a hold-out window.
