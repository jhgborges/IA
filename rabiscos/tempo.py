import pandas as pd
import matplotlib.pyplot as plt

# Importe o dataset
df = pd.read_csv('weather_data_bangladesh.csv')

# Examine os dados
print(df.head())

# Análise da temperatura máxima

# Plote a temperatura máxima ao longo do tempo
plt.plot(df['date'], df['max_temp'])
plt.xlabel('Data')
plt.ylabel('Temperatura máxima (°C)')
plt.show()

# Análise da temperatura mínima

# Plote a temperatura mínima ao longo do tempo
plt.plot(df['date'], df['min_temp'])
plt.xlabel('Data')
plt.ylabel('Temperatura mínima (°C)')
plt.show()

# Análise das precipitações

# Plote as precipitações ao longo do tempo
plt.plot(df['date'], df['rain'])
plt.xlabel('Data')
plt.ylabel('Precipitações (mm)')
plt.show()

# Análise das médias mensais de temperatura máxima

# Calcule as médias mensais de temperatura máxima
monthly_max_temp = df.groupby('month')['max_temp'].mean()

# Plote as médias mensais de temperatura máxima
plt.plot(monthly_max_temp.index, monthly_max_temp.values)
plt.xlabel('Mês')
plt.ylabel('Temperatura máxima (°C)')
plt.show()

# Análise das médias mensais de temperatura mínima

# Calcule as médias mensais de temperatura mínima
monthly_min_temp = df.groupby('month')['min_temp'].mean()

# Plote as médias mensais de temperatura mínima
plt.plot(monthly_min_temp.index, monthly_min_temp.values)
plt.xlabel('Mês')
plt.ylabel('Temperatura mínima (°C)')
plt.show()

# Análise das médias mensais de precipitações

# Calcule as médias mensais de precipitações
monthly_rain = df.groupby('month')['rain'].mean()

# Plote as médias mensais de precipitações
plt.plot(monthly_rain.index, monthly_rain.values)
plt.xlabel('Mês')
plt.ylabel('Precipitações (mm)')
plt.show()
