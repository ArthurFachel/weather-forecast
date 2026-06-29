"""
owner: @ArthurFachel
linkedin: https://www.linkedin.com/in/arthur-fachel-801218176/
discord: lampejo__
"""

"""
Weather repo analysis: cleans the raw CSV, runs EDA, flags anomalies,
forecasts temperature for one location, and pulls out a few extra
country/climate breakdowns.

Run order matters since later sections reuse `df` from the cleaning step.
Charts and tables get dumped into OUTPUT_DIR as they're made.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "./GlobalWeatherRepository.csv"
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})


print("[1] Loading raw data ...")
df = pd.read_csv(DATA_PATH)
print(f"     Raw shape: {df.shape}")

# pull calendar features out of the timestamp, used later for monthly /
# hourly breakdowns
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
df["year"] = df["last_updated"].dt.year
df["month"] = df["last_updated"].dt.month
df["day"] = df["last_updated"].dt.day
df["hour"] = df["last_updated"].dt.hour
df["day_of_week"] = df["last_updated"].dt.dayofweek

# everything we treat as a number, force-cast so bad strings become NaN
# instead of blowing up downstream
num_cols = [
    "temperature_celsius",
    "temperature_fahrenheit",
    "wind_kph",
    "wind_degree",
    "pressure_mb",
    "pressure_in",
    "precip_mm",
    "precip_in",
    "humidity",
    "cloud",
    "feels_like_celsius",
    "feels_like_fahrenheit",
    "visibility_km",
    "visibility_miles",
    "uv_index",
    "gust_kph",
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# no temp reading = nothing to do with the row, so drop instead of impute
initial_len = len(df)
df = df.dropna(subset=["temperature_celsius"]).copy()
print(f"     After dropping missing target rows: {len(df)} ({initial_len - len(df)} removed)")

# same country/location/timestamp/temp = same reading logged twice
df = df.drop_duplicates(subset=["country", "location_name", "last_updated", "temperature_celsius"])
print(f"     After dedup: {len(df)}")

# fill gaps with that country's median first, global median if the
# country has nothing to go on
for c in num_cols:
    if c not in df.columns:
        continue
    df[c] = df.groupby("country")[c].transform(lambda x: x.fillna(x.median()))
    df[c] = df[c].fillna(df[c].median())

# clip (don't drop) outliers past 1.5*IQR so sensor glitches can't skew
# stats/models too hard
for c in ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "pressure_mb", "gust_kph"]:
    if c not in df.columns:
        continue
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    df[c] = df[c].clip(low, high)

# z-scored copies of a few features, kept around in case something
# downstream wants scaled input (not used by the tree/linear models below)
scaler = StandardScaler()
scaled_features = ["temperature_celsius", "humidity", "wind_kph", "pressure_mb", "precip_mm", "cloud", "uv_index"]
for c in scaled_features:
    if c not in df.columns:
        continue
    df[f"{c}_scaled"] = scaler.fit_transform(df[[c]].values)


print("[2] EDA ...")

# summary_statistics.txt
summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.txt")
with open(summary_path, "w") as f:
    f.write(df[num_cols].describe().to_string())
print(f"     Summary stats -> {summary_path}")

# temperature_distribution.png: histograms for the 6 main metrics
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
plot_dict = {
    "temperature_celsius": "Temperature (C)",
    "humidity": "Humidity (%)",
    "precip_mm": "Precipitation (mm)",
    "wind_kph": "Wind Speed (kph)",
    "pressure_mb": "Pressure (mb)",
    "cloud": "Cloud Cover (%)",
}
for ax, (col, title) in zip(axes.ravel(), plot_dict.items()):
    if col in df.columns:
        sns.histplot(df[col], kde=True, ax=ax, bins=60)
        ax.set_title(title)
        ax.set_xlabel(title)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "temperature_distribution.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# correlation_heatmap.png: lower triangle only, upper half is redundant
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=False, square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title("Pearson Correlation Heatmap (Numerical Features)")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# temp_time_series.png: global daily mean temp
df["date"] = df["last_updated"].dt.date
daily = df.groupby("date")["temperature_celsius"].mean().reset_index()
daily["date"] = pd.to_datetime(daily["date"])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily["date"], daily["temperature_celsius"], color="steelblue", linewidth=0.8)
ax.set_title("Global Mean Temperature Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Mean Temperature (°C)")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "temp_time_series.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# temperature_by_country.png: boxplot, top 20 countries by row count
top20 = df["country"].value_counts().nlargest(20).index.tolist()
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(data=df[df["country"].isin(top20)], x="country", y="temperature_celsius", ax=ax, order=top20)
ax.set_title("Temperature Distribution by Country (Top 20 by Volume)")
ax.set_xlabel("Country")
ax.set_ylabel("Temperature (°C)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "temperature_by_country.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# precipitation_patterns.png
if "precip_mm" in df.columns:
    monthly_precip = df.groupby("month")["precip_mm"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=monthly_precip, x="month", y="precip_mm", palette="Blues_d", ax=ax)
    ax.set_title("Average Precipitation by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Precipitation (mm)")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "precipitation_patterns.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"     Saved {save_path}")

# geographical_temperature_map.png: just raw lat/lon as x/y, no
# projection/basemap, but gets the point across
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(
    df["longitude"], df["latitude"], c=df["temperature_celsius"], cmap="coolwarm", s=3, alpha=0.6
)
ax.set_title("Geographical Temperature Scatter")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Temperature (°C)")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "geographical_temperature_map.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")


print("[3] Advanced EDA")

# Isolation Forest flags the weirdest ~2% of temp/humidity/pressure/wind
# combos as anomalies (anomaly == -1), plotted on temp vs humidity
model_features = ["temperature_celsius", "humidity", "pressure_mb", "wind_kph"]
available = [c for c in model_features if c in df.columns]
iso_df = df[available].copy()
iso = IsolationForest(n_estimators=150, contamination=0.02, random_state=42, n_jobs=5)
iso_df["anomaly"] = iso.fit_predict(iso_df)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["red" if x == -1 else "steelblue" for x in iso_df["anomaly"]]
ax.scatter(iso_df["temperature_celsius"], iso_df["humidity"], c=colors, s=4, alpha=0.6)
ax.set_title("Anomaly Detection (Isolation Forest) — Temperature vs Humidity")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "anomaly_detection.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# air quality vs weather corr: only use pollutant columns with >100
# non-null values, otherwise it's noise
aq_cols = [
    "air_quality_PM2.5", "air_quality_PM10", "air_quality_Carbon_Monoxide",
    "air_quality_Ozone", "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
]
weather_cols = ["temperature_celsius", "humidity", "wind_kph", "pressure_mb", "precip_mm"]
available_aq = [c for c in aq_cols if c in df.columns and df[c].notna().sum() > 100]
if available_aq:
    aq_weather = df[available_aq + weather_cols].corr().loc[available_aq, weather_cols]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(aq_weather, annot=True, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
    ax.set_title("Air Quality vs Weather Parameters Correlation")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "air_quality_correlation.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
else:
    print("Skipped air-quality correlation (insufficient data).")


print("[4] Forecasting Models ...")

# forecast on whichever single location has the most readings — keeps
# the series to one consistent climate instead of a blended mess
location_counts = df.groupby("location_name").size().sort_values(ascending=False)
top_loc = location_counts.index[0]
print(f"Using top location by record count: {top_loc} ({location_counts.iloc[0]} records)")

loc_df = df[df["location_name"] == top_loc].copy()
loc_df = loc_df.sort_values("last_updated").reset_index(drop=True)

# collapse to one row per day
daily_loc = loc_df.groupby("date").agg({
    "temperature_celsius": "mean",
    "humidity": "mean",
    "wind_kph": "mean",
    "pressure_mb": "mean",
    "precip_mm": "mean",
}).reset_index()
daily_loc["date"] = pd.to_datetime(daily_loc["date"])
daily_loc = daily_loc.sort_values("date").reset_index(drop=True)

target_col = "temperature_celsius"
feature_cols = ["humidity", "wind_kph", "pressure_mb", "precip_mm"]
for c in feature_cols:
    if c not in daily_loc.columns:
        feature_cols.remove(c)

# top location doesn't have enough days on its own (<14) — fall back to
# a global daily series so the models have something to chew on
if len(daily_loc) < 14:
    print("WARNING: insufficient daily data for forecasting; using global daily instead.")
    daily_loc = df.groupby("date").agg({
        "temperature_celsius": "mean",
        "humidity": "mean",
        "wind_kph": "mean",
        "pressure_mb": "mean",
        "precip_mm": "mean",
    }).reset_index()
    daily_loc["date"] = pd.to_datetime(daily_loc["date"])
    daily_loc = daily_loc.sort_values("date").reset_index(drop=True)


# days_since_start gives the models a linear time index, lag1 (yesterday's
# temp) doubles as both a feature and the naive baseline prediction
daily_loc["days_since_start"] = (daily_loc["date"] - daily_loc["date"].min()).dt.days
daily_loc["lag1"] = daily_loc[target_col].shift(1)

# first row has no lag1, drop it along with anything else missing
daily_loc = daily_loc.dropna(subset=[target_col, "lag1"] + feature_cols).reset_index(drop=True)

print(f"Daily rows for forecast: {len(daily_loc)}")

# chronological split (no shuffling) — last 15% of days held out so the
# test set is genuinely "the future" relative to training
split_idx = int(len(daily_loc) * 0.85)
train = daily_loc.iloc[:split_idx].copy()
test = daily_loc.iloc[split_idx:].copy()

X_train = train[["days_since_start", "lag1"] + feature_cols]
X_test = test[["days_since_start", "lag1"] + feature_cols]
y_train = train[target_col]
y_test = test[target_col]

results = {}

# naive baseline: just predict today = yesterday. anything fancier needs
# to actually beat this
naive_pred = test["lag1"].values
results["Naive"] = {
    "MAE": mean_absolute_error(y_test, naive_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, naive_pred)),
    "MAPE": np.mean(np.abs((y_test - naive_pred) / y_test)) * 100,
    "R2": r2_score(y_test, naive_pred),
}
print(f"Naive baseline done")

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results["LinearRegression"] = {
    "MAE": mean_absolute_error(y_test, lr_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
    "MAPE": np.mean(np.abs((y_test - lr_pred) / y_test)) * 100,
    "R2": r2_score(y_test, lr_pred),
}
print(f"Linear Regression done")

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=5)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results["RandomForest"] = {
    "MAE": mean_absolute_error(y_test, rf_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
    "MAPE": np.mean(np.abs((y_test - rf_pred) / y_test)) * 100,
    "R2": r2_score(y_test, rf_pred),
}
print(f"Random Forest done")

xgb = xgboost.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
    random_state=42, eval_metric="rmse", n_jobs=5,
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
results["XGBoost"] = {
    "MAE": mean_absolute_error(y_test, xgb_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)),
    "MAPE": np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100,
    "R2": r2_score(y_test, xgb_pred),
}
print(f"XGBoost done")

# weekly-seasonal SARIMA first (temp only, univariate), drop down to a
# plain ARIMA(2,1,2) if SARIMA chokes on this series
train_ts = train[target_col]
try:
    sarima = SARIMAX(train_ts, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=False)
    sarima_pred = sarima_fit.forecast(steps=len(test))
except Exception:
    from statsmodels.tsa.arima.model import ARIMA
    arima = ARIMA(train_ts, order=(2, 1, 2))
    arima_fit = arima.fit()
    sarima_pred = arima_fit.forecast(steps=len(test))

results["ARIMA/SARIMA"] = {
    "MAE": mean_absolute_error(y_test, sarima_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, sarima_pred)),
    "MAPE": np.mean(np.abs((y_test - sarima_pred) / y_test)) * 100,
    "R2": r2_score(y_test, sarima_pred),
}
print(f"     ARIMA/SARIMA done")

# blend everything, weighted by each model's own R2 (floored at 0.01 so
# nothing gets a zero weight). Naive/ARIMA only get a vote if their R2 is
# actually positive, otherwise they'd just drag the blend down
preds = {
    "LinearRegression": lr_pred,
    "RandomForest": rf_pred,
    "XGBoost": xgb_pred,
    "Naive": naive_pred,
    "ARIMA/SARIMA": sarima_pred.values if hasattr(sarima_pred, "values") else np.array(sarima_pred),
}
pos = {k: v for k, v in results.items() if v["R2"] > 0 or k in ["LinearRegression", "RandomForest", "XGBoost"]}
weights = {}
for k in pos:
    r2 = results[k]["R2"]
    w = max(r2, 0.01)
    weights[k] = w

total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}

ensemble_pred = np.zeros(len(y_test))
for k, w in weights.items():
    ensemble_pred += preds[k] * w

results["Ensemble"] = {
    "MAE": mean_absolute_error(y_test, ensemble_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_pred)),
    "MAPE": np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100,
    "R2": r2_score(y_test, ensemble_pred),
}
print(f"     Ensemble done")

# model_comparison.txt / .png
comp = pd.DataFrame(results).T
comp = comp[["MAE", "RMSE", "MAPE", "R2"]]
comp_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")
with open(comp_path, "w") as f:
    f.write(comp.to_string())
print(f"     Model comparison table -> {comp_path}")

fig, ax = plt.subplots(figsize=(10, 5))
comp_reset = comp.reset_index().rename(columns={"index": "Model"})
comp_melt = comp_reset.melt(id_vars="Model", value_vars=["MAE", "RMSE", "MAPE"])

sns.barplot(data=comp_melt, x="Model", y="value", hue="variable", ax=ax)
ax.set_title("Model Comparison (MAE / RMSE / MAPE)")
ax.set_xlabel("Model")
ax.set_ylabel("Error")
ax.legend(title="Metric")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# actual_vs_predicted.png, best single model (XGBoost) next to the ensemble
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, xgb_pred, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0].set_title(f"XGBoost: Actual vs Predicted (R²={results['XGBoost']['R2']:.3f})")
axes[0].set_xlabel("Actual Temperature (°C)")
axes[0].set_ylabel("Predicted Temperature (°C)")

axes[1].scatter(y_test, ensemble_pred, alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[1].set_title(f"Ensemble: Actual vs Predicted (R²={results['Ensemble']['R2']:.3f})")
axes[1].set_xlabel("Actual Temperature (°C)")
axes[1].set_ylabel("Predicted Temperature (°C)")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")


print("[5] Unique Analyses ...")

# climate_trends.png, 30-day rolling mean temp, top 5 countries by volume
# min_periods=1 means the first ~29 days still get a value (using
# whatever history exists yet) instead of coming out as NaN
climate = df.sort_values("last_updated").copy()
climate["date"] = pd.to_datetime(climate["last_updated"].dt.date)

top5_countries = df["country"].value_counts().nlargest(5).index.tolist()
climate_top5 = climate[climate["country"].isin(top5_countries)].copy()
climate_agg = climate_top5.groupby(["country", "date"])["temperature_celsius"].mean().reset_index()
climate_agg = climate_agg.sort_values(["country", "date"])
climate_agg["rolling_30"] = climate_agg.groupby("country")["temperature_celsius"].transform(lambda x: x.rolling(30, min_periods=1).mean())

fig, ax = plt.subplots(figsize=(14, 6))
for country in top5_countries:
    sub = climate_agg[climate_agg["country"] == country]
    ax.plot(sub["date"], sub["rolling_30"], label=country)
ax.set_title("30-Day Rolling Mean Temperature (Top 5 Countries)")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "climate_trends.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# feature_importance.png, trained on a 30k-row sample for speed.
# comparing RF's built-in importance against permutation importance since
# the built-in one can be biased toward high-cardinality features
sample_df = df.sample(n=min(30000, len(df)), random_state=42).copy()
ml_features = ["humidity", "wind_kph", "pressure_mb", "precip_mm", "cloud", "uv_index", "hour", "month", "latitude", "longitude"]
available_ml = [c for c in ml_features if c in sample_df.columns and sample_df[c].notna().sum() > 0]
X_all = sample_df[available_ml].fillna(sample_df[available_ml].median())
y_all = sample_df[target_col].fillna(sample_df[target_col].median())

rf2 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=5)
rf2.fit(X_all, y_all)

perm = permutation_importance(rf2, X_all, y_all, n_repeats=5, random_state=42, n_jobs=5)

importances = pd.DataFrame({
    "Feature": available_ml,
    "RF_Importance": rf2.feature_importances_,
    "Permutation_Importance": perm.importances_mean,
})
importances = importances.sort_values("RF_Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(importances))
width = 0.35
ax.bar(x - width/2, importances["RF_Importance"], width, label="RF Importance")
ax.bar(x + width/2, importances["Permutation_Importance"], width, label="Permutation Importance")
ax.set_xticks(x)
ax.set_xticklabels(importances["Feature"], rotation=45, ha="right")
ax.set_title("Feature Importance: Random Forest vs Permutation")
ax.legend()
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

# spatial_analysis_latitude_bands.png — these bands are a rough custom
# split, not official climate-zone boundaries
if "latitude" in df.columns:
    df["lat_band"] = pd.cut(
        df["latitude"],
        bins=[-np.inf, -23.5, 23.5, 35, 60, np.inf],
        labels=["Arctic_Antarctic", "Temperate_S", "Tropical", "Temperate_N", "Subarctic"],
    )
    spatial = df.groupby("lat_band")["temperature_celsius"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=spatial, x="lat_band", y="mean", ax=ax, palette="coolwarm")
    ax.set_title("Mean Temperature by Latitude Band")
    ax.set_xlabel("Latitude Band")
    ax.set_ylabel("Mean Temperature (°C)")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "spatial_analysis_latitude_bands.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"     Saved {save_path}")

# geographical_patterns.txt/.png — hottest/coldest/wettest/driest/windiest
# countries. each column is ranked independently, rows don't line up
# across columns
country_agg = df.groupby("country").agg({
    "temperature_celsius": "mean",
    "precip_mm": "mean",
    "wind_kph": "mean",
}).reset_index().dropna()

patterns = pd.DataFrame({
    "Hottest": country_agg.nlargest(10, "temperature_celsius")["country"].values,
    "Coldest": country_agg.nsmallest(10, "temperature_celsius")["country"].values,
    "Wettest": country_agg.nlargest(10, "precip_mm")["country"].values,
    "Driest": country_agg.nsmallest(10, "precip_mm")["country"].values,
    "Windiest": country_agg.nlargest(10, "wind_kph")["country"].values,
})

patterns_path = os.path.join(OUTPUT_DIR, "geographical_patterns.txt")
with open(patterns_path, "w") as f:
    f.write(patterns.to_string(index=False))
print(f"     Geographical patterns -> {patterns_path}")

fig, ax = plt.subplots(figsize=(12, 6))
top20_hot = country_agg.nlargest(20, "temperature_celsius")
sns.barplot(data=top20_hot, x="country", y="temperature_celsius", palette="YlOrRd", ax=ax)
ax.set_title("Top 20 Hottest Countries (Mean Temperature)")
ax.set_xlabel("Country")
ax.set_ylabel("Mean Temperature (°C)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "geographical_patterns.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"     Saved {save_path}")

print("\n All outputs generated successfully in:", OUTPUT_DIR)