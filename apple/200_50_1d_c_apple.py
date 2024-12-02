import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar datos del activo Apple (AAPL), utilizando el período máximo disponible
data = yf.download('AAPL', period='max', interval='1d')

# Calcular la media móvil exponencial rápida (50 días) y lenta (200 días)
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Crear las señales de compra y cierre de posiciones (sin posiciones en corto)
data['Signal'] = 0
data.iloc[50:, data.columns.get_loc('Signal')] = np.where(data['EMA50'][50:] > data['EMA200'][50:], 1, 0)

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
plt.figure(figsize=(10, 5))
plt.plot(data['Cumulative Market Return'], label='Market Return')
plt.plot(data['Cumulative Strategy Return'], label='Strategy Return')
plt.title('Estrategia de Backtesting de Medias Móviles Exponenciales (EMA50 y EMA200) - Solo Compras')
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
sharpe_ratio = data['Strategy'].mean() / data['Strategy'].std() * np.sqrt(252)  # 252 es el número de días de trading
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')