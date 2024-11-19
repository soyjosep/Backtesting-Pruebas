import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# (El resto del código permanece igual hasta la sección donde ocurre el error)

# Crear un DataFrame con los retornos de la estrategia para todos los tickers
df_estrategia = pd.concat(estrategia_retornos, axis=1)

# Eliminar columnas con datos insuficientes
df_estrategia = df_estrategia.dropna(axis=1, how='all')

# Calcular el retorno promedio por periodo (suponiendo igual peso)
df_estrategia['Retorno_Promedio'] = df_estrategia.mean(axis=1)

# Eliminar filas con NaN en 'Retorno_Promedio'
df_estrategia = df_estrategia.dropna(subset=['Retorno_Promedio'])

# Verificar el tipo de índice
print(f"Tipo de índice antes del resampleo: {type(df_estrategia.index)}")

# Asegurarnos de que el índice es un DatetimeIndex
if not isinstance(df_estrategia.index, pd.DatetimeIndex):
    df_estrategia.index = pd.to_datetime(df_estrategia.index)

# Resamplear los retornos a frecuencia diaria
df_estrategia_diario = df_estrategia['Retorno_Promedio'].resample('D').sum()

# Continuar con el resto del código sin cambios
# Calcular el retorno acumulado total de la estrategia
df_estrategia_diario = df_estrategia_diario.to_frame()
df_estrategia_diario['Estrategia_Acumulada_Total'] = (1 + df_estrategia_diario['Retorno_Promedio']).cumprod()

# Calcular el beneficio total en porcentaje
beneficio_total = (df_estrategia_diario['Estrategia_Acumulada_Total'].iloc[-1] - 1) * 100

# Calcular el beneficio/pérdida promedio diario en porcentaje
beneficio_diario_promedio = df_estrategia_diario['Retorno_Promedio'].mean() * 100

# Mostrar los resultados
fecha_inicio = df_estrategia_diario.index[0].date()
fecha_fin = df_estrategia_diario.index[-1].date()
print("\nResultados del Backtesting:")
print(f"Beneficio total de la estrategia combinada utilizando EMAs de 73 y 312 periodos en gráfico de 1 hora: {beneficio_total:.2f}%")
print(f"Beneficio/Pérdida promedio diario: {beneficio_diario_promedio:.5f}%")
print(f"Período de análisis: {fecha_inicio} - {fecha_fin}")