# 4. Observable universe

A classifier fit on earlier sessions is applied to later ones. That works only for a column whose later values fall in the range seen earlier. A tree cannot extrapolate: a value above everything in the training rows lands in the same leaf as the largest training value. A column measured in reais or in shares grows with the company, so its later values leave the earlier range and the column marks the calendar period instead of the next session.

The screen reads the development file only. It splits development into thirds of 1,241 sessions. The first third runs from 2007-07-03 through 2013-03-04. The last third runs from 2018-11-28 through 2023-12-28. For each candidate $x$, let $q_{0.01}$ and $q_{0.99}$ be the 1st and 99th percentiles over the first third. The outside share is

$$s = \frac{1}{n_3}\sum_{t \in \text{last third}} \mathbf{1}\{x_t < q_{0.01} \ \text{or}\ x_t > q_{0.99}\}.$$

A column with a stable distribution leaves about 2 percent of the last third outside that range. A candidate enters the observable universe when $s \le 0.1$, five times that rate.

| Feature | First-third median | Last-third median | Outside share | Screen |
| --- | --- | --- | --- | --- |
| CloseLag1 | 1.96 | 29.36 | 100.0% | Excluded |
| CloseLag2 | 1.959 | 29.36 | 100.0% | Excluded |
| CloseLag3 | 1.959 | 29.36 | 100.0% | Excluded |
| HMLLag1 | 0.05536 | 0.7427 | 91.5% | Excluded |
| HMLLag2 | 0.05522 | 0.7427 | 91.5% | Excluded |
| HMLLag3 | 0.05504 | 0.7427 | 91.9% | Excluded |
| OMCLag1 | $9.92 \times 10^{-4}$ | 0.008939 | 74.3% | Excluded |
| OMCLag2 | $9.86 \times 10^{-4}$ | 0.008939 | 74.5% | Excluded |
| OMCLag3 | $9.72 \times 10^{-4}$ | 0.008939 | 74.5% | Excluded |
| VolumeLag1 | $2.05 \times 10^{6}$ | $7.55 \times 10^{6}$ | 16.2% | Excluded |
| VolumeLag2 | $2.04 \times 10^{6}$ | $7.55 \times 10^{6}$ | 16.2% | Excluded |
| VolumeLag3 | $2.04 \times 10^{6}$ | $7.55 \times 10^{6}$ | 18.4% | Excluded |
| CloseEMA2 | 1.957 | 29.42 | 100.0% | Excluded |
| CloseEMA4 | 1.956 | 29.43 | 100.0% | Excluded |
| CloseEMA8 | 1.962 | 29.46 | 100.0% | Excluded |
| Ret1 | 0 | $4.93 \times 10^{-4}$ | 2.4% | Kept |
| Ret21 | 0.0134 | 0.01481 | 6.5% | Kept |
| Mom12_1 | 0.1359 | 0.2148 | 8.9% | Kept |
| RVol21 | 0.01975 | 0.01929 | 2.3% | Kept |
| Parkinson21 | $4.32 \times 10^{-4}$ | $3.55 \times 10^{-4}$ | 6.0% | Kept |
| LogVolChg1 | $4.70 \times 10^{-4}$ | -0.017 | 0.0% | Kept |
| Amihud21 | $4.60 \times 10^{-9}$ | $6.60 \times 10^{-11}$ | 100.0% | Excluded |
| Range1 | 0.02895 | 0.02783 | 1.1% | Kept |
| Intraday1 | $-5.15 \times 10^{-4}$ | $-5.40 \times 10^{-4}$ | 2.2% | Kept |
| Overnight1 | 0.001009 | 0.001433 | 2.0% | Kept |

9 of 25 candidates pass: Ret1, Ret21, Mom12_1, RVol21, Parkinson21, LogVolChg1, Range1, Intraday1, Overnight1. The 16 excluded columns are CloseLag1, CloseLag2, CloseLag3, HMLLag1, HMLLag2, HMLLag3, OMCLag1, OMCLag2, OMCLag3, VolumeLag1, VolumeLag2, VolumeLag3, CloseEMA2, CloseEMA4, CloseEMA8, Amihud21. Every excluded column is measured in reais, in shares, or, for Amihud21, per real traded. Each scale-free counterpart that measures the same quantity passes the screen: Ret1 for the lagged close, Range1 for high minus low, Intraday1 for open minus close, and LogVolChg1 for volume. The limit of 10 percent and the two percentiles are the only settings of the screen.
