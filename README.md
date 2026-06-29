# Weather Trend Forecasting 

## Dataset
- **File:** GlobalWeatherRepository.csv 
- **Source:** Kaggle: World Weather Repository (nelgiriyewithana/global-weather-repository)
- **Rows/Cols:** ~150K rows, 40+ features (temperature, humidity, wind, air quality, location, time)

### 1. Data Cleaning & Preprocessing
- Handle missing values (impute or drop)
- Handle outliers (IQR or z-score)
- Normalize/scale numerical features
- Parse `last_updated` as datetime, extract time features (year, month, day, hour, day_of_week)
- Deduplicate if needed

### 2. Exploratory Data Analysis (EDA)
- Summary statistics for all numerical features
- Distribution plots: temperature_celsius, humidity, precipitation, wind_kph, pressure
- Correlation heatmap (Pearson) of all numerical features
- Time series plot of temperature by country/region over time
- Box plots of temperature by country (top 20 countries)
- Precipitation patterns by month/season
- Geographical scatter: latitude/longitude colored by temperature

### 3. Advanced EDA
- **Anomaly Detection:** Isolation Forest or IQR-based. Visualize anomalies on temp distribution.
- **Correlation Analysis:** Strongest correlations between weather params and air quality.

### 4. Forecasting Models (Multi-Model + Ensemble)
Use `last_updated` as time axis. Aggregate (mean) per day for multiple records.

Models (target: temperature_celsius):
- **Baseline:** Naive (persist last value)
- **ARIMA/SARIMA:** Seasonal time series
- **Linear Regression:** time features + lags
- **Random Forest:** non-linear baseline
- **XGBoost:** gradient boosting
- **Ensemble:** weighted avg of top 3

Metrics: MAE, RMSE, MAPE, R^2. Comparison table. Actual vs predicted plot.

### 5. Unique Analyses
- **Climate Analysis:** Group by country/continent, compute rolling means
- **Environmental Impact:** Correlate air quality (PM2.5, PM10, CO, Ozone, NO2, SO2) with temp/humidity/wind
- **Feature Importance:** RF feature_importances_ + Permutation Importance, side by side
- **Spatial Analysis:** Latitude bands (tropical/ temperate/ arctic) — analyze per band
- **Geographical Patterns:** Aggregate by country — hottest/coldest/wettest/driest/windiest. Map visualizations.


### Output Files (all to `output/`)
- correlation_heatmap.png
- temperature_distribution.png
- temp_time_series.png
- temperature_by_country.png
- anomaly_detection.png
- model_comparison.png
- actual_vs_predicted.png
- feature_importance.png
- climate_trends.png
- air_quality_correlation.png
- geographical_temperature_map.png
- precipitation_patterns.png

### How to Run

- `python weather_forecast.py` 

