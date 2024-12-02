import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Descargar datos de Bitcoin (BTC-USD), utilizando el período máximo disponible con un intervalo de 1 hora
data = yf.download('BTC-USD', period='max', interval='1h')

# Calcular la media móvil exponencial rápida (8 horas) y lenta (83 horas)
data['EMA8'] = data['Close'].ewm(span=8, adjust=False).mean()
data['EMA83'] = data['Close'].ewm(span=83, adjust=False).mean()

# Crear las señales de compra y cierre de posiciones (sin posiciones en corto)
data['Signal'] = 0
data.iloc[83:, data.columns.get_loc('Signal')] = np.where(data['EMA8'][83:] > data['EMA83'][83:], 1, 0)

# Calcular las posiciones basadas en las señales (comprar o estar fuera del mercado)
data['Position'] = data['Signal']

# Calcular el rendimiento horario
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
plt.title('Estrategia de Backtesting de Medias Móviles Exponenciales (EMA8 y EMA83) - Solo Compras (Bitcoin)')
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
sharpe_ratio = data['Strategy'].mean() / data['Strategy'].std() * np.sqrt(252 * 24)  # 252 días, 24 horas
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')