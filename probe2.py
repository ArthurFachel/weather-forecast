import pandas as pd, json, sys

df = pd.read_csv('/home/fachel/projects/weather-forecast/GlobalWeatherRepository.csv')
print('Shape:', df.shape)
print('Date range raw:', df['last_updated'].dropna().min(), df['last_updated'].dropna().max())
# parse dates
df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
print('After parse min/max:', df['last_updated'].min(), df['last_updated'].max())
print('Unique dates count:', df['last_updated'].dt.date.nunique())
print('Unique countries count:', df['country'].nunique())
