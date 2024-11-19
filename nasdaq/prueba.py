import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from concurrent.futures import ThreadPoolExecutor

# Obtener la lista actualizada de las empresas del NASDAQ 100
def get_nasdaq_100_tickers():
    tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA', 'PYPL', 'ADBE',
        # Agrega aquí todos los tickers del NASDAQ 100 actualizados
    ]
    return tickers

# Función para descargar datos históricos de 1 hora para un ticker
def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)
        if data.empty:
            print(f"No se encontraron datos para {ticker}")
            return None
        data['Ticker'] = ticker

        # Asegurar que las columnas sean de un solo nivel
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data
    except Exception as e:
        print(f"Error al descargar datos para {ticker}: {e}")
        return None

# Preparar los datos para cada ticker
def prepare_data(tickers):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=729)  # Últimos 730 días

    all_data = []

    # Descargar datos en paralelo
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_data, ticker, start_date, end_date) for ticker in tickers]
        for future in futures:
            data = future.result()
            if data is not None and not data.empty:
                all_data.append(data)

    # Alinear los datos al menor rango común
    if not all_data:
        print("No se pudo descargar datos para ningún ticker.")
        return [], None, None

    latest_start_date = max([df.index[0] for df in all_data])
    earliest_end_date = min([df.index[-1] for df in all_data])

    for i in range(len(all_data)):
        all_data[i] = all_data[i][(all_data[i].index >= latest_start_date) & (all_data[i].index <= earliest_end_date)]

    return all_data, latest_start_date, earliest_end_date

# Función de backtesting
def backtest(all_data, start_ma, end_ma):
    best_profit = float('-inf')
    best_fast_ma = None
    best_slow_ma = None

    # Reducir el rango de medias móviles para optimizar el tiempo de ejecución
    max_ma = min(100, end_ma)  # Ajusta este valor según tus necesidades

    # Iterar a través de todas las combinaciones de medias móviles rápidas y lentas
    for fast_ma in range(start_ma, max_ma + 1):
        for slow_ma in range(fast_ma + 1, max_ma + 1):
            total_profit = 0
            for data in all_data:
                # Asegurar que 'Close' es una Serie y no está vacía
                if 'Close' not in data.columns or data['Close'].empty:
                    continue  # Saltar si no hay datos de cierre

                # Extraer 'Close' como una Serie única
                close_series = data['Close']

                # Verificar si 'close_series' es realmente una Serie
                if isinstance(close_series, pd.DataFrame):
                    # Si es un DataFrame, intenta aplanarlo a una Serie
                    close_series = close_series.iloc[:, 0]

                # Asegurarse de que 'close_series' es una Serie y no está vacía
                if not isinstance(close_series, pd.Series) or close_series.empty:
                    continue

                close_series = close_series.astype(float).copy()

                # Calcular las medias móviles
                fast_ma_series = close_series.rolling(window=fast_ma).mean()
                slow_ma_series = close_series.rolling(window=slow_ma).mean()

                # Crear un DataFrame con los resultados usando pd.concat
                df = pd.concat([close_series, fast_ma_series, slow_ma_series], axis=1)

                # Verificar que la concatenación resultó en exactamente 3 columnas
                if df.shape[1] != 3:
                    print(f"Advertencia: El DataFrame concatenado tiene {df.shape[1]} columnas en lugar de 3. Ticker: {data['Ticker'].iloc[0]}")
                    continue

                df.columns = ['Close', 'Fast_MA', 'Slow_MA']
                df.dropna(inplace=True)

                # Verificar si df no está vacío después de dropna
                if df.empty:
                    continue  # Saltar si no hay suficientes datos

                # Calcular posiciones y señales
                df['Position'] = np.where(df['Fast_MA'] > df['Slow_MA'], 1, 0)
                df['Signal'] = df['Position'].diff()

                # Precios de compra y venta
                buy_prices = df[df['Signal'] == 1]['Close']
                sell_prices = df[df['Signal'] == -1]['Close']

                # Asegurarse de que haya igual número de señales de compra y venta
                min_len = min(len(buy_prices), len(sell_prices))
                buy_prices = buy_prices.iloc[:min_len]
                sell_prices = sell_prices.iloc[:min_len]

                # Calcular las ganancias
                profits_list = sell_prices.values - buy_prices.values
                total_profit += profits_list.sum()

            # Verificar y actualizar el mejor rendimiento
            if total_profit > best_profit:
                best_profit = total_profit
                best_fast_ma = fast_ma
                best_slow_ma = slow_ma

    return best_fast_ma, best_slow_ma, best_profit

def main():
    tickers = get_nasdaq_100_tickers()
    all_data, start_date, end_date = prepare_data(tickers)

    if not all_data:
        print("No hay datos suficientes para realizar el backtesting.")
        return

    print(f"Datos alineados desde {start_date} hasta {end_date}")

    best_fast_ma, best_slow_ma, best_profit = backtest(all_data, 1, 1000)

    print(f"Las mejores medias móviles son Fast MA: {best_fast_ma}, Slow MA: {best_slow_ma}")
    print(f"Beneficio Total: {best_profit}")

if __name__ == "__main__":
    main()