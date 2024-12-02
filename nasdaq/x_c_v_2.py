import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Función para obtener datos históricos de Binance
def get_historical_data(symbol, interval, start_str, end_str=None):
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000  # Límite máximo de registros por solicitud
    data = []
    start_time = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_time = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'limit': limit
        }
        if end_time:
            params['endTime'] = end_time

        response = requests.get(url, params=params)
        temp_data = response.json()

        if not temp_data or 'code' in temp_data:
            break

        data.extend(temp_data)
        start_time = temp_data[-1][0] + 1

        if len(temp_data) < limit or (end_time and start_time >= end_time):
            break

        time.sleep(0.5)  # Espera para no exceder el límite de la API

    return data

# Obtener datos históricos de los últimos 2 años
symbol = 'BTCUSDT'
interval = '1h'
end_time = datetime.now()
start_time = end_time - timedelta(days=730)

data = get_historical_data(
    symbol,
    interval,
    start_time.strftime('%Y-%m-%d %H:%M:%S'),
    end_time.strftime('%Y-%m-%d %H:%M:%S')
)

# Convertir los datos en un DataFrame
columns = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close_time', 'Quote_asset_volume', 'Number_of_trades',
    'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
]

df = pd.DataFrame(data, columns=columns)

# Convertir tipos de datos
df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(float)

# Ordenar y resetear el índice
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Definir función para procesar cada valor de n
def process_n(n):
    total_profit = 0
    position = None  # 'long' o 'short'
    entry_price = None
    entry_date = None
    trades = []  # Lista para almacenar las operaciones realizadas

    # Comenzamos con una posición neutra y buscamos la primera entrada
    for i in range(n, len(df)):
        current_high = df.loc[i, 'High']
        current_low = df.loc[i, 'Low']
        current_close = df.loc[i, 'Close']
        current_date = df.loc[i, 'Date']
        prev_highs = df.loc[i-n:i-1, 'High']
        prev_lows = df.loc[i-n:i-1, 'Low']

        max_prev_high = prev_highs.max()
        min_prev_low = prev_lows.min()

        if position is None:
            # Decidir si entrar en posición larga o corta
            if current_high > max_prev_high:
                position = 'long'
                entry_price = current_close
                entry_date = current_date
            elif current_low < min_prev_low:
                position = 'short'
                entry_price = current_close
                entry_date = current_date
        else:
            # Verificar si debemos cambiar de posición
            if position == 'long':
                if current_low < min_prev_low:
                    # Cerrar posición larga y abrir posición corta
                    exit_price = current_close
                    exit_date = current_date
                    profit = exit_price - entry_price
                    total_profit += profit
                    # Registrar la operación
                    trades.append({
                        'n': n,
                        'position': 'long',
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    # Abrir posición corta
                    position = 'short'
                    entry_price = current_close
                    entry_date = current_date
            elif position == 'short':
                if current_high > max_prev_high:
                    # Cerrar posición corta y abrir posición larga
                    exit_price = current_close
                    exit_date = current_date
                    profit = entry_price - exit_price
                    total_profit += profit
                    # Registrar la operación
                    trades.append({
                        'n': n,
                        'position': 'short',
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    # Abrir posición larga
                    position = 'long'
                    entry_price = current_close
                    entry_date = current_date

    # Al finalizar, cerramos cualquier posición abierta
    if position is not None:
        exit_price = df.loc[len(df)-1, 'Close']
        exit_date = df.loc[len(df)-1, 'Date']
        if position == 'long':
            profit = exit_price - entry_price
        elif position == 'short':
            profit = entry_price - exit_price
        total_profit += profit
        # Registrar la operación
        trades.append({
            'n': n,
            'position': position,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit
        })

    # Devolver el resultado para este valor de n
    return {
        'n': n,
        'total_profit': total_profit,
        'trades': trades
    }

# Preparar los valores de n
n_values = list(range(1, 1001))

# Ejecutar el procesamiento en paralelo con barras de progreso
if __name__ == '__main__':
    # Número de procesos (núcleos de CPU)
    num_processes = cpu_count()

    # Crear un pool de procesos
    with Pool(processes=num_processes) as pool:
        # Mapear la función con tqdm para mostrar progreso
        results = list(tqdm(pool.imap_unordered(process_n, n_values), total=len(n_values)))

    # Convertimos los resultados en un DataFrame para facilitar la manipulación
    results_df = pd.DataFrame([{'n': res['n'], 'total_profit': res['total_profit']} for res in results])

    # Ordenamos los resultados por el beneficio total en orden descendente
    results_df.sort_values('total_profit', ascending=False, inplace=True)

    # Mostramos el top 10 de las mejores combinaciones
    top_10 = results_df.head(10)
    print("Top 10 combinaciones (Operaciones Largas y Cortas con Condiciones Simétricas):")
    print(top_10)

    # Si deseas ver las operaciones de la mejor combinación
    best_n = top_10.iloc[0]['n']
    # Encontrar el resultado correspondiente a best_n
    for res in results:
        if res['n'] == best_n:
            best_trades = res['trades']
            break

    # Convertir las operaciones en un DataFrame
    trades_df = pd.DataFrame(best_trades)

    print(f"\nOperaciones para n = {best_n}:")
    print(trades_df)