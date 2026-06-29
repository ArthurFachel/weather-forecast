
import pandas as pd, json, sys

df = pd.read_csv('/home/fachel/projects/weather-forecast/GlobalWeatherRepository.csv')
print('Shape:', df.shape)
print('Columns:', json.dumps(list(df.columns), indent=0))
print(df.head(3).to_string(index=False))
