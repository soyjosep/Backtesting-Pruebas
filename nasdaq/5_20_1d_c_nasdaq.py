import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Función para obtener la lista de tickers del NASDAQ 100
def obtener_tickers_nasdaq100():
    url = 'https://es.wikipedia.org/wiki/NASDAQ-100'
    tablas = pd.read_html(url)
    for i, tabla in enumerate(tablas):
        if 'Ticker' in tabla.columns:
            tickers = tabla['Ticker'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers
        elif 'Symbol' in tabla.columns:
            tickers = tabla['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers
    return []

# Función para obtener el rango común de fechas entre todos los tickers
def obtener_rango_comun_fechas(tickers):
    fechas_inicio = []
    fechas_fin = []
    tickers_validos = []
    for ticker in tickers:
        try:
            print(f"Obteniendo rango de fechas para {ticker}...")
            data = yf.download(ticker, interval='1d', period='max', progress=False)
            if data.empty:
                print(f"No se pudo obtener datos para {ticker}.")
                continue
            fechas_inicio.append(data.index[0])
            fechas_fin.append(data.index[-1])
            tickers_validos.append(ticker)
        except Exception as e:
            print(f"Error al procesar {ticker}: {e}")
            continue
    if not fechas_inicio or not fechas_fin:
        raise Exception("No se pudieron obtener fechas para ningún ticker.")
    fecha_inicio_comun = max(fechas_inicio)
    fecha_fin_comun = min(fechas_fin)
    print(f"\nRango común de fechas: {fecha_inicio_comun.date()} - {fecha_fin_comun.date()}")
    return fecha_inicio_comun, fecha_fin_comun, tickers_validos

# Diccionario para almacenar los retornos de la estrategia de cada ticker
estrategia_retornos = {}

# Función para realizar el backtesting para un ticker específico
def backtesting_medias_moviles_exponenciales(ticker, fecha_inicio, fecha_fin):
    try:
        print(f"Descargando datos para {ticker} desde {fecha_inicio.date()} hasta {fecha_fin.date()}...")
        data = yf.download(tickers=ticker, interval='1d', start=fecha_inicio, end=fecha_fin, progress=False)
        print(f"Datos descargados para {ticker}: {len(data)} filas.")
        if data.empty:
            print(f"No se pudo obtener datos para {ticker}.")
            return

        # Calcular medias móviles exponenciales de 5 y 20 periodos
        data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

        # Generar señales de compra (1) y venta (-1)
        data['Signal'] = 0
        data.loc[data['EMA5'] > data['EMA20'], 'Signal'] = 1
        data.loc[data['EMA5'] < data['EMA20'], 'Signal'] = -1

        # Calcular retornos diarios
        data['Retornos'] = data['Close'].pct_change()

        # Calcular retornos de la estrategia
        data['Estrategia'] = data['Signal'].shift(1) * data['Retornos']

        # Almacenar los retornos de la estrategia
        estrategia_retornos[ticker] = data['Estrategia']

    except Exception as e:
        print(f"Error al procesar {ticker}: {e}")

# Obtener lista de tickers
tickers = obtener_tickers_nasdaq100()

# Limitar el número de tickers para agilizar el proceso (opcional)
# tickers = tickers[:10]

# Obtener el rango común de fechas y la lista actualizada de tickers válidos
fecha_inicio_comun, fecha_fin_comun, tickers_validos = obtener_rango_comun_fechas(tickers)

# Iterar sobre cada ticker válido y realizar el backtesting
for ticker in tickers_validos:
    print(f"Procesando {ticker}...")
    backtesting_medias_moviles_exponenciales(ticker, fecha_inicio_comun, fecha_fin_comun)

# Crear un DataFrame con los retornos de la estrategia para todos los tickers
df_estrategia = pd.DataFrame(estrategia_retornos)

# Calcular el retorno promedio diario de la estrategia (suponiendo igual peso)
df_estrategia['Retorno_Promedio'] = df_estrategia.mean(axis=1)

# Eliminar filas con NaN en 'Retorno_Promedio'
df_estrategia = df_estrategia.dropna(subset=['Retorno_Promedio'])

# Calcular el retorno acumulado total de la estrategia
df_estrategia['Estrategia_Acumulada_Total'] = (1 + df_estrategia['Retorno_Promedio']).cumprod()

# Calcular el beneficio total en porcentaje
beneficio_total = (df_estrategia['Estrategia_Acumulada_Total'].iloc[-1] - 1) * 100

# Calcular el beneficio/pérdida diaria promedio en porcentaje
beneficio_diario_promedio = df_estrategia['Retorno_Promedio'].mean() * 100

# Mostrar los resultados
print("\nResultados del Backtesting:")
print(f"Beneficio total de la estrategia combinada utilizando EMAs de 5 y 20 periodos: {beneficio_total:.2f}%")
print(f"Beneficio/Pérdida diaria promedio: {beneficio_diario_promedio:.5f}%")
print(f"Período de análisis: {fecha_inicio_comun.date()} - {fecha_fin_comun.date()}")