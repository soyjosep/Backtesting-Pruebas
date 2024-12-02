import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar datos de Bitcoin (BTC-USD) utilizando el período máximo disponible
data = yf.download('BTC-USD', period='max', interval='1d')

# Calcular la media móvil exponencial rápida (2 días) y lenta (16 días)
data['EMA2'] = data['Close'].ewm(span=2, adjust=False).mean()
data['EMA16'] = data['Close'].ewm(span=16, adjust=False).mean()

# Crear las señales de compra y cierre de posiciones (sin posiciones en corto)
data['Signal'] = 0
data.iloc[16:, data.columns.get_loc('Signal')] = np.where(data['EMA2'][16:] > data['EMA16'][16:], 1, 0)

# Calcular las posiciones basadas en las señales (comprar o estar fuera del mercado)
data['Position'] = data['Signal']

# Calcular el rendimiento diario
data['Return'] = np.log(data['Close'] / data['Close'].shift(1))

# Calcular el rendimiento de la estrategia (solo en posiciones largas)
data['Strategy'] = data['Return'] * data['Position'].shift(1)

# Acumular retornos
data['Cumulative Market Return'] = data['Return'].cumsum().apply(np.exp)
data['Cumulative Strategy Return'] = data['Strategy'].cumsum().apply(np.exp)

# Visualizar el rendimiento de la estrategia frente al mercado
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Cumulative Market Return'], label='Retorno del Mercado')
plt.plot(data.index, data['Cumulative Strategy Return'], label='Retorno de la Estrategia')
plt.title('Estrategia de EMAs (2 y 16 días) en Bitcoin - Solo Compras')
plt.xlabel('Fecha')
plt.ylabel('Retorno Acumulado')
plt.legend()
plt.show()

# Calcular el drawdown máximo
def max_drawdown(returns):
    cum_returns = returns.cumsum().apply(np.exp)
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# Drawdown de la estrategia
drawdown = max_drawdown(data['Strategy'])
print(f'Max Drawdown: {drawdown * 100:.2f}%')

# Calcular el Sharpe Ratio
sharpe_ratio = data['Strategy'].mean() / data['Strategy'].std() * np.sqrt(252)  # 252 es el número de días de trading al año
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')