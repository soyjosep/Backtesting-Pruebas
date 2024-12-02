import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Obtener datos históricos de Bitcoin utilizando yfinance
symbol = 'BTC-USD'  # Símbolo de Bitcoin en Yahoo Finance

# Definir la fecha de inicio y fin
start_date = '2014-09-17'  # Fecha inicial disponible en yfinance para BTC-USD
end_date = datetime.now().strftime('%Y-%m-%d')  # Fecha actual

# Descargar los datos históricos
df = yf.download(symbol, start=start_date, end=end_date, interval='1d')

# Verificar si se han obtenido datos
if df.empty:
    print("No se han obtenido datos. Por favor, verifica el rango de fechas y el símbolo.")
    exit()

# Resetear el índice para que 'Date' sea una columna
df.reset_index(inplace=True)

# Si las columnas son MultiIndex, las simplificamos
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Imprimir información sobre el DataFrame
print("Shape del DataFrame:", df.shape)
print("Columnas del DataFrame:", df.columns)
print(df.head())

# Eliminar filas con datos faltantes
df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

# Ordenar y resetear el índice
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Definir función para procesar cada valor de n
def process_n(n):
    total_profit = 0
    position = None  # No estamos en una posición al inicio
    trades = []  # Lista para almacenar las operaciones realizadas

    # Iterar sobre el DataFrame desde el n-ésimo índice hasta el final
    for i in range(n, len(df)):
        current_high = df.loc[i, 'High']
        current_low = df.loc[i, 'Low']
        current_close = df.loc[i, 'Close']
        current_date = df.loc[i, 'Date']
        prev_highs = df.loc[i-n:i-1, 'High']
        prev_lows = df.loc[i-n:i-1, 'Low']

        max_prev_high = prev_highs.max()
        min_prev_low = prev_lows.min()

        # Si no estamos en una posición, buscamos una entrada
        if position is None:
            # Señal para posición larga
            if current_high > max_prev_high:
                # Entramos en una posición larga al precio de cierre
                entry_price = current_close
                entry_date = current_date
                position = {
                    'type': 'long',
                    'entry_price': entry_price,
                    'entry_index': i,
                    'entry_date': entry_date
                }
            # Señal para posición corta
            elif current_low < min_prev_low:
                # Entramos en una posición corta al precio de cierre
                entry_price = current_close
                entry_date = current_date
                position = {
                    'type': 'short',
                    'entry_price': entry_price,
                    'entry_index': i,
                    'entry_date': entry_date
                }
        else:
            # Si estamos en una posición, verificamos si debemos salir
            if position['type'] == 'long':
                # Señal para salir de posición larga
                if current_low < min_prev_low:
                    # Salimos de la posición larga al precio de cierre
                    exit_price = current_close
                    exit_date = current_date
                    profit = exit_price - position['entry_price']
                    total_profit += profit
                    # Registrar la operación
                    trades.append({
                        'n': n,
                        'type': 'long',
                        'entry_date': position['entry_date'],
                        'exit_date': exit_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    position = None  # Reseteamos la posición
            elif position['type'] == 'short':
                # Señal para salir de posición corta
                if current_high > max_prev_high:
                    # Salimos de la posición corta al precio de cierre
                    exit_price = current_close
                    exit_date = current_date
                    profit = position['entry_price'] - exit_price
                    total_profit += profit
                    # Registrar la operación
                    trades.append({
                        'n': n,
                        'type': 'short',
                        'entry_date': position['entry_date'],
                        'exit_date': exit_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'profit': profit
                    })
                    position = None  # Reseteamos la posición

    # Si terminamos y aún estamos en una posición, la cerramos al último precio de cierre
    if position is not None:
        exit_price = df.loc[len(df)-1, 'Close']
        exit_date = df.loc[len(df)-1, 'Date']
        if position['type'] == 'long':
            profit = exit_price - position['entry_price']
        elif position['type'] == 'short':
            profit = position['entry_price'] - exit_price
        total_profit += profit
        # Registrar la operación
        trades.append({
            'n': n,
            'type': position['type'],
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
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
    print("Top 10 combinaciones (Operaciones Largas y Cortas):")
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