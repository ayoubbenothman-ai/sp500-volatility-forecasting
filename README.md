# Forecasting 21-Day Realized Volatility of the S&P 500

Comparison of naive persistence, a HAR-style regression, GARCH and ML Regression Models (LR & SVR) for forecasting the realized volatility of the
S&P 500 index 21-days ahead.

**[Full technical note (PDF)](volatility_forecasting_note.pdf)** ·
**[Notebook](sp500-volatility-forecasting.ipynb)**

## Question

Can machine-learning regressions improve out-of-sample forecasts of index
volatility relatively to persistence and classical volatility models 
at a 21-trading-day horizon?

The target at date *t* is the annualized standard deviation of the **next** 21
daily log returns. Predictors use returns up to *t* only.

## Evaluation design

Two properties of the target drive the design: it is a noisy proxy for latent
volatility, and consecutive targets overlap by 20 of their 21 returns.

- **Walk-forward, expanding window.** The model is refitted every 21 trading days
  and predicts the following block.
- **21-day embargo.** Training stops 21 days before the forecast origin, so no
  training target overlaps the test block.
- **Selection / test split.** Forecasts before `SPLIT="2017-01-11"` are used for diagnostics
  and hyperparameter choice. The later period is used once, for the results below.
- **Non-overlapping scoring.** Only the first forecast of each block is kept.
- **QLIKE on variances** as the primary loss, alongside RMSE, R² and R² relative
  to the naive and HAR benchmarks.

## Results

<!-- TODO: paste the final table from the notebook -->

| Model | QLIKE | RMSE | R² | R² vs naive | R² vs HAR | Corr |
|---|---:|---:|---:|---:|---:|---:|
| GARCH(1,1) | **0.6607** | 0.1044 | 0.1409 | −0.0148 | −0.0968 | 0.5863 |
| HAR-style | 0.7535 | 0.0997 | 0.2167 | 0.0748 | 0.0000 | 0.5919 |
| OLS | 0.8223 | 0.0962 | 0.2707 | 0.1385 | 0.0689 | **0.5982** |
| Naive | 0.8514 | 0.1036 | 0.1534 | 0.0000 | −0.0808 | 0.5784 |
| SVR (selected) | 0.8613 | **0.0956** | **0.2792** | **0.1486** | **0.0798** | 0.5961 |

*82 non-overlapping 21-day forecasts, 2017-01-11 to 2023-10-16. Sorted by QLIKE (lower is better); best value per column in bold.*

**The ranking depends on the loss.**
GARCH(1,1) has the lowest QLIKE, yet its squared error is essentially that of
naive persistence ($R^2_{\text{vs naive}} = -0.015$). Conversely, SVR and OLS
have the lowest RMSE and the highest $R^2$, but QLIKE values close to
persistence (SVR is in fact the worst of the five on this metric). All five models have almost the same correlation with realized
volatility (0.58--0.60): they contain similar information and differ mainly in
the \emph{level and asymmetry} of their errors.

![Forecast vs realized volatility](figures/realized-ols-garch.png)

## Running it

```bash
git clone https://github.com/<user>/sp500-volatility-forecasting.git
cd sp500-volatility-forecasting
pip install -r requirements.txt
jupyter lab notebooks/sp500_volatility_forecasting.ipynb
```

The notebook downloads its own data from Yahoo Finance, so no data files are
included. Runtime is a approximately 5 minutes. The SVR hyperparameter search is the slowest
section.

## Limitations
- From 2017-01-11 to 2023-10-16 is about 6.76 years, roughly 1,703 trading days at 252 per year, and 1703 / 21 ≈ 81. 81 forecasts is maybe too few to evaluate models.

## Repository layout

```
notebooks/   analysis notebook
report/      technical note.
figures/     figures used by the note and README.
```

## Author
Ayoub Ben Othman
Email: ayoub.ben-othman@polytechnique.edu

