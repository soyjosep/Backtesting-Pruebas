import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Supongamos que ya tienes el dataframe 'data' cargado con tus precios de Bitcoin
# El dataframe debe tener al menos una columna llamada 'Close' con los precios de cierre.

# Cálculo de las medias móviles (5 y 20 períodos)
data['SMA5'] = data['Close'].rolling(window=5).mean()
data['SMA20'] = data['Close'].rolling(window=20).mean()

# Señales de compra/venta
data['Signal'] = 0  # Creamos una nueva columna para almacenar las señales
data['Signal'][5:] = np.where(data['SMA5'][5:] > data['SMA20'][5:], 1, 0)  # Señal de compra cuando la SMA5 cruza por encima de SMA20
data['Position'] = data['Signal'].diff()  # Para identificar los cambios en la posición

# Visualización de la estrategia en el gráfico
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Precio de Cierre', alpha=0.5)
plt.plot(data['SMA5'], label='SMA5', alpha=0.75)
plt.plot(data['SMA20'], label='SMA20', alpha=0.75)

# Señales de compra (flechas verdes) y venta (flechas rojas)
plt.plot(data[data['Position'] == 1].index, 
         data['SMA5'][data['Position'] == 1], 
         '^', markersize=10, color='g', lw=0, label='Compra')
plt.plot(data[data['Position'] == -1].index, 
         data['SMA5'][data['Position'] == -1], 
         'v', markersize=10, color='r', lw=0, label='Venta')

plt.title('Estrategia de Cruce de Medias Móviles (SMA5 y SMA20)')
plt.legend(loc='best')
plt.show()

# Backtesting simple
initial_balance = 10000  # Supongamos que empezamos con $10,000
balance = initial_balance
btc_held = 0

for i in range(len(data)):
    if data['Position'][i] == 1:  # Señal de compra
        if balance > 0:
            btc_held = balance / data['Close'][i]
            balance = 0
    elif data['Position'][i] == -1:  # Señal de venta
        if btc_held > 0:
            balance = btc_held * data['Close'][i]
            btc_held = 0

# Valor final al final del periodo de backtesting
if btc_held > 0:
    balance = btc_held * data['Close'].iloc[-1]  # Si aún tenemos Bitcoin, lo convertimos a cash al precio final

profit = balance - initial_balance
print(f'Ganancia final: ${profit:.2f}')