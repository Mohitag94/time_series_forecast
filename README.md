# Time Series Forecasting with ARIMA and LSTM/GRU

A comparative case study benchmarking a classical statistical model (ARIMA) against deep learning recurrent architectures (LSTM, GRU, hybrid LSTM-GRU) on two real-world time series datasets, originally completed as part of the *Advanced Research Topics in Data Science* module (MSc Data Science, University of Hertfordshire).

## Problem

Given a historical time series, which forecasting approach performs better: a classical statistical model or a deep learning model? This project answers that question empirically on two datasets with different characteristics — a slow-moving, seasonal sales series and a volatile daily stock price series — to see whether the answer depends on the nature of the data.

**Datasets:**
- **Johnson & Johnson quarterly sales** — January 1960 to October 1980, 84 entries, recorded every 3 months
- **Amazon daily closing stock price** — February 2018 to February 2023, 1,259 entries

**Forecast horizon:** 24 months for both datasets

## Methodology

1. **Exploratory analysis** — checked for null values, examined autocorrelation (ACF) and partial autocorrelation (PACF) to inform ARIMA's `p` parameter, and decomposed each series to check for trend and seasonality.
2. **Stationarity testing** — used the Augmented Dickey-Fuller (ADF) test (H0: series has a unit root / is non-stationary). Applied Box-Cox transformation to stabilise variance, then differencing to achieve stationarity.
3. **ARIMA modelling** — selected `(p, d, q)` orders via the Box-Jenkins method, minimising AIC across a grid search of `p` and `q` (range 1–8). Forecasting used a rollback method: the model is retrained after each new forecasted step is folded back in as if it were observed data.
4. **Recurrent neural networks** — trained LSTM, GRU, and a hybrid LSTM-GRU model. Hyperparameters (including number of layers) were tuned using Bayesian optimisation via Keras Tuner. Features were engineered from lagged values (lag count set to match ARIMA's `p`), seasonal/trend/residual decomposition, and moving averages. Data was scaled with Min-Max/MinMax scaling.
5. **Fourier analysis** — applied as a supplementary technique to examine the frequency-domain structure of both series and compare their dominant periodic components.

## Results

### Johnson & Johnson Sales — ARIMA order (6, 2, 1)
Box-Cox did not fully stabilise the series (λ ≈ 0.0507); second-order differencing achieved stationarity (ADF p-value ≈ 0.0061, rejecting H0 at 5%).

| Model | MSE | MAE | RMSE |
|---|---|---|---|
| LSTM | 0.004542 | 0.058575 | 0.067398 |
| **GRU** | **0.000431** | **0.020210** | **0.020764** |
| LSTM-GRU | 0.004051 | 0.059436 | 0.063649 |

### Amazon Stock Price — ARIMA order (2, 1, 2)
Box-Cox transformation (λ ≈ −0.37) did not achieve stationarity; first-order differencing did (ADF p-value ≈ 0.0, rejecting H0 at 5%). Highly correlated pricing variables (open/high/low/close/adj. close) were dropped, keeping only Close and Volume.

| Model | MSE | MAE | RMSE |
|---|---|---|---|
| LSTM | 0.001777 | 0.036803 | 0.042159 |
| **GRU** | **0.000480** | **0.016904** | **0.021914** |
| LSTM-GRU | 0.001482 | 0.032616 | 0.038502 |

GRU produced the lowest error by every metric on **both** datasets.

## Key finding — metric performance isn't the whole story

Lowest-error-wins is the headline result, but the case study's conclusion is more nuanced, and it's the more interesting part:

- **For Johnson & Johnson**, despite ARIMA's higher error metrics than GRU, a visual comparison of the forecasts showed **ARIMA actually captured the underlying trend and seasonality better** than the GRU model. This is a useful reminder that MSE alone doesn't always tell the full story — a model can post a lower error while still failing to capture the structure that actually matters for a forecast to be useful in practice.
- **For Amazon's stock price**, none of the models — ARIMA or the RNN variants — captured the series' underlying pattern well. This isn't surprising: stock prices are driven substantially by unpredictable, non-seasonal volatility, and the forecasts degrade into flat or oscillating output the further out they project, which is visible in the forecast plots.
- The **ARIMA multivariate extension (VARMAX)** was attempted on the Amazon dataset but was computationally infeasible on available hardware (excessive runtime, exhausted RAM); resampling the data was also tried without satisfactory results.

## Fourier analysis

A supplementary frequency-domain comparison of the two series found:
- J&J's sales series has sparse, evenly-spaced frequency components consistent with its quarterly reporting cycle, and the Fourier reconstruction closely tracked the original series.
- Amazon's daily price series has dense, numerous frequency components with a smoother power spectrum, and its Fourier reconstruction captured the broad trend (including the pandemic-era surge and correction) while smoothing over daily volatility.

## What I'd improve next

- Incorporate multivariate features (e.g. trading volume alongside price) more effectively, rather than dropping correlated variables outright
- Explore additional feature engineering for the Amazon series specifically — rolling standard deviation, median, variance, and inverse-Fourier noise filtering, as noted in the original case study as a promising next step
- Evaluate forecasts using a metric that rewards structural pattern-matching (trend/seasonality capture), not just point-wise error, given the gap observed between ARIMA's MSE and its qualitative performance on the J&J data

## Tech stack

Python, statsmodels (ARIMA), TensorFlow/Keras, Keras Tuner (Bayesian optimisation), pandas, NumPy

## How to run

```
git clone https://github.com/Mohitag94/time_series_forecast.git
cd time_series_forecast
pip install -r requirements.txt
```

*(Add a `requirements.txt` if one doesn't already exist, and confirm this matches how your notebook(s) are actually organised in the repo.)*
