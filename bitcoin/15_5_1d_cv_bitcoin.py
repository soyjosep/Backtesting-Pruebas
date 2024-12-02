import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar datos de Bitcoin (BTC-USD), utilizando el período máximo disponible con un intervalo de 1 día
data = yf.download('BTC-USD', period='max', interval='1d')

# Calcular la media móvil exponencial rápida (5 días) y lenta (15 días)
data['EMA5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA15'] = data['Close'].ewm(span=15, adjust=False).mean()

# Crear las señales de compra y venta
data['Signal'] = 0
data['Signal'] = np.where(data['EMA5'] > data['EMA15'], 1, np.where(data['EMA5'] < data['EMA15'], -1, 0))

# Asegurar que las posiciones se ajustan al día siguiente de la señal
data['Position'] = data['Signal'].shift(1)

# Calcular el rendimiento diario
data['Return'] = data['Close'].pct_change()

# Calcular el rendimiento de la estrategia
data['Strategy'] = data['Return'] * data['Position']

# Acumular retornos
data['Cumulative Market Return'] = (1 + data['Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy']).cumprod()

# Visualizar el precio de Bitcoin junto con el rendimiento de la estrategia
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='Precio de Bitcoin')
plt.title('Precio de Bitcoin')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data.index, data['Cumulative Market Return'], label='Retorno de Mercado Acumulado')
plt.plot(data.index, data['Cumulative Strategy Return'], label='Retorno de la Estrategia Acumulado')
plt.title('Comparación del Retorno de Mercado y Estrategia')
plt.legend()
plt.tight_layout()
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
sharpe_ratio = data['Strategy'].mean() / data['Strategy'].std() * np.sqrt(252)  # 252 es el número de días de trading
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')