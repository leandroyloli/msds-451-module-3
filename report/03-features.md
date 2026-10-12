# 3. Features

The target is the direction of the close-to-close log return on WEGE3.SA. On day $t$,

$$r_t = \log(C_t / C_{t-1}), \qquad y_t = \mathbf{1}\{r_t > 0\}.$$

Each of the 25 candidates is known at the close of day $t-1$. The open, high, low, close, and volume of day $t$ do not enter any candidate. Values are stored at full precision.

The 25 columns are the candidate pool. Chapter 4 screens them for drift before any selection. The first 15 follow the course example and are measured in reais or in shares. The other 10 come from the literature. Nine of them are returns, ratios, or log changes, which do not grow with the price level. Amihud21 is measured per real traded. The adjusted close of WEG rises from about 0.09 reais in 2000 to about 55 reais in 2026, so a lag of the price mostly tracks that rise.

## Definitions

The three price averages are exponentially weighted means of closes through day $t-1$. The half-life is $\tau$, the smoothing weight is $\alpha$, and observation $i$ steps back has weight $w_i$:

$$\alpha = 1 - \exp(-\ln 2 / \tau), \qquad w_i = (1 - \alpha)^i.$$

CloseEMA2, CloseEMA4, and CloseEMA8 use $\tau = 1$, $2$, and $4$. The names follow the course example. The weights are normalized, which is the adjusted form of the moving average.

| Feature | Definition | Role |
| --- | --- | --- |
| CloseLag1 | $C_{t-1}$ | Example. Yesterday's price level. |
| CloseLag2 | $C_{t-2}$ | Example. Price level two sessions back. |
| CloseLag3 | $C_{t-3}$ | Example. Price level three sessions back. |
| HMLLag1 | $H_{t-1} - L_{t-1}$ | Example. Yesterday's range, in reais. |
| HMLLag2 | $H_{t-2} - L_{t-2}$ | Example. |
| HMLLag3 | $H_{t-3} - L_{t-3}$ | Example. |
| OMCLag1 | $O_{t-1} - C_{t-1}$ | Example. Yesterday's open-to-close change, in reais. |
| OMCLag2 | $O_{t-2} - C_{t-2}$ | Example. |
| OMCLag3 | $O_{t-3} - C_{t-3}$ | Example. |
| VolumeLag1 | $V_{t-1}$ | Example. Undefined when the recorded volume is zero. |
| VolumeLag2 | $V_{t-2}$ | Example. Same zero-volume rule. |
| VolumeLag3 | $V_{t-3}$ | Example. Same zero-volume rule. |
| CloseEMA2 | weighted mean above, $\tau = 1$ | Example. |
| CloseEMA4 | weighted mean above, $\tau = 2$ | Example. |
| CloseEMA8 | weighted mean above, $\tau = 4$ | Example. |
| Ret1 | $\log(C_{t-1}/C_{t-2})$ | One-day reversal (Jegadeesh, 1990; Lehmann, 1990). |
| Ret21 | $\log(C_{t-1}/C_{t-22})$ | About one month of past return. |
| Mom12_1 | $\log(C_{t-21}/C_{t-252})$ | Twelve-month return, skipping the most recent month (Jegadeesh and Titman, 1993). |
| RVol21 | $\sqrt{\frac{1}{20}\sum_{s=t-21}^{t-1}(r_s-\bar r)^2}$ | Recent volatility. Sample standard deviation of 21 daily log returns. |
| Parkinson21 | $\frac{1}{21}\sum_{s=t-21}^{t-1}\frac{[\log(H_s/L_s)]^2}{4\ln 2}$ | Range-based variance (Parkinson, 1980). It uses the high and the low. |
| LogVolChg1 | $\log(V_{t-1}/V_{t-2})$ | Volume shock, on positive volume (Campbell, Grossman, and Wang, 1993). |
| Amihud21 | $\frac{1}{21}\sum_{s=t-21}^{t-1}\frac{\lvert r_s\rvert}{C_s V_s}$ | Illiquidity (Amihud, 2002). Each session in the window must have positive volume. |
| Range1 | $(H_{t-1}-L_{t-1})/C_{t-1}$ | Yesterday's range as a fraction of price. Scale-free HML. |
| Intraday1 | $\log(C_{t-1}/O_{t-1})$ | Yesterday's open-to-close return. Scale-free OMC. |
| Overnight1 | $\log(O_{t-1}/C_{t-2})$ | Yesterday's overnight gap. |

## Volume and the estimation sample

Yahoo records volume of zero on 1,814 of the 6,721 downloaded sessions, including nearly every session from 2000 through 2006. A zero is treated as missing volume, not as a session with no shares traded. Volume lags, the log volume change, and the Amihud ratio are undefined on a zero print. A row is kept only when all 25 candidates are defined. Closes from before the first kept row still enter the moving averages and Mom12_1. They are history, not rows of their own. No missing value is filled in.

The kept table has 4,418 sessions, from 3 July 2007 through 9 October 2026. Of these, 2,229 (50.5 percent) are up days and 2,189 are flat or down. Within that span, 368 downloaded sessions are absent because a zero-volume print leaves at least one candidate undefined.

The same file stores the day's raw prices, the same-day range, the same-day open-minus-close, and the log return, so the candidates can be checked. Those columns are not part of the 25.

RSI, MACD, Bollinger bands, fractional differentiation, and any series that is not in this file of daily open, high, low, close, and volume are outside the candidate list. Chapter 2 cuts this table into a development window and a hold-out window.
