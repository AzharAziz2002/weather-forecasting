# Global Weather Trend Forecasting (Advanced Assessment)

> **PM Accelerator mission:** By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.

Analysis and forecasting of the Kaggle **Global Weather Repository** (daily weather for cities worldwide, 40+ features).
The full generated report (with all figures) is `outputs/report.md`.

## What this project does
| Requirement | Where |
|---|---|
| Data cleaning (missing values, outliers, normalisation) | `load_and_clean` - per-location median imputation, 1.5xIQR outlier counts, 1st/99th percentile winsorising, z-score scaling |
| EDA + temperature/precipitation visuals | `eda` |
| **Anomaly detection** | `anomalies` - Isolation Forest (multivariate) + local z-score (location x month) |
| **Multiple forecasting models + ensemble** | `forecasting` - naive, seasonal-naive, harmonic ridge, SARIMAX (Fourier exog), damped ETS, Random Forest and Gradient Boosting on lags; ensembles: average, inverse-RMSE weighted, top-3 |
| Metrics | MAE, RMSE, MAPE, sMAPE, R2 on a 14-day hold-out (`last_updated` is the time axis) |
| Climate analysis | `climate` - continent x month climatology and seasonal range by country (trend analysis runs only when 24+ months of consistent data exist) |
| Environmental impact | `air_quality` - pollutants vs weather correlations, PM2.5 by continent/country |
| Feature importance | `feature_importance` - Pearson, Spearman, mutual information, RF impurity, permutation + consensus rank |
| Spatial analysis | `spatial` - lon/lat maps of temperature, precipitation, PM2.5, anomalies |
| Geographical patterns | `geography` - continent/country/hemisphere comparisons |

## Method notes
- **Forecast target:** global daily mean temperature, averaged over a stable panel of locations observed on at least 90% of days (so cities joining the repository do not create artificial jumps). The last 14 days are the test set; the 14 days before are validation, used *only* to compute ensemble weights. Models are refit on train+validation before testing (no leakage). A final 14-day future forecast with an approximate 95% band is written to `outputs/future_forecast.csv`.
- **Data-quality finding:** the UTC time at which observations were recorded drifted during collection (about 14:00 in 2024 to about 06:00 in 2026), which shifts temperatures by several degrees. The pipeline detects this from `last_updated_epoch` and runs the forecasting, seasonal/climate, country and map analyses only on rows recorded within +/-3 h of the latest observation time, from the first month after which at least 90% of rows qualify (section 1b of the report; change the tolerance with `--time-tol`). Countries with fewer than 100 rows are ignored in country rankings (a few stray one-day records have localised country names).
- **Continent** is derived from the `timezone` column prefix (no extra dependency).
- **Caveats:** (1) after the time-of-day filter only about 17 months of consistent data remain, so long-term climate *trends* cannot be estimated (the report says so and analyses the seasonal cycle and regional differences instead); (2) all locations are observed at the same UTC time, i.e. different local times of day, so country/continent comparisons partly reflect local time; (3) the model comparison uses a single 14-day test window, so small differences between top models are not conclusive.

## Run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download GlobalWeatherRepository.csv from Kaggle into data/
python src/weather_analysis.py --data data/GlobalWeatherRepository.csv --out outputs
```
Takes a few minutes. Outputs: `outputs/report.md` (full report with figures), `outputs/figures/*.png`, `model_metrics.csv`, `feature_importance.csv`, `country_summary.csv`, `top_anomalies.csv`, `future_forecast.csv`.

Smoke test without Kaggle data: `python src/make_sample_data.py data/sample_weather.csv` (synthetic, same schema) then run with `--data data/sample_weather.csv`.
Diagnostic helpers used to find the time-of-day issue: `src/diagnose.py`, `src/diagnose_hours.py`, `src/diagnose_utc.py`.

## Results
- **Data:** 168,785 rows, 268 locations, 2024-05-16 to 2026-10-02. No missing values in the raw file (the cleaning code handles them anyway).
- **Data-quality finding:** the UTC time of observations drifted from ~14:00 (2024) to ~06:00 (2026), shifting temperatures by several degrees. Forecasting, seasonal, country and map analyses use only rows within +/-3 h of 06:00 UTC from 2025-06 onward (489 days, 185 stable locations).
- **Forecasting (14-day hold-out):** best model `ensemble_top3` RMSE 0.338 C / MAE 0.278 C; best single model SARIMAX RMSE 0.345 C; naive baseline RMSE 0.929 C. The top models are close to each other on a 14-day window.
- **Anomalies:** Isolation Forest flags 1% of records (most in Indonesia); a local z-score flags unusual temperatures per location and month.
- **Climate:** seasonal swing is largest in Mongolia (42.6 C). Trends are not estimated because only ~17 months of consistent data exist.
- **Air quality:** PM2.5 is lower on rainy days (9.6 vs 15.3) and most associated with humidity (Spearman -0.26); Saudi Arabia has the highest median PM2.5.
- **Feature importance:** latitude, UV index and pressure are the top predictors of temperature (RF hold-out R2 0.93).
- **Geography:** temperature falls with distance from the equator (r = -0.72); hemispheres have opposite seasons.
- **Limitations:** all locations are observed at the same UTC time, so country comparisons partly reflect local time of day.

## Demo video
Add your 1-2 minute screen recording link here.
