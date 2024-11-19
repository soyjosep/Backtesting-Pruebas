import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time
import os

# Ignorar advertencias de pandas para una salida más limpia
warnings.filterwarnings("ignore")

def get_nasdaq_100_tickers():
    tickers = [
        'ADBE', 'AMD', 'A', 'ABNB', 'ALGN', 'ALLE', 'GOOGL', 'GOOG',
        'AMZN', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'ASML', 'ADSK', 'ADP',
        'BIDU', 'BIIB', 'BMRN', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR',
        'CHKP', 'CTAS', 'CSCO', 'CTSH', 'CMCSA', 'COP', 'CPRT', 'COST',
        'CRWD', 'DOCU', 'DLTR', 'EBAY', 'EA', 'EXC', 'EXPE', 'FAST',
        'META', 'FISV', 'FOXA', 'FOX', 'GILD', 'HAS', 'HSIC', 'HLT',
        'HMC', 'INTU', 'JD', 'KLAC', 'KHC', 'LRCX', 'LBTYA', 'LBTYK',
        'MAR', 'MELI', 'MSFT', 'MRNA', 'MDLZ', 'NTES', 'NFLX', 'NVDA',
        'ORLY', 'PAYX', 'PYPL', 'PEP', 'QCOM', 'REGN', 'ROST', 'SPGI',
        'CRM', 'SGEN', 'SIRI', 'SWKS', 'SPLK', 'SBUX', 'TTWO', 'TMUS',
        'TSLA', 'TXN', 'ISRG', 'VRSN', 'VRTX', 'WBA', 'WDC', 'WDAY',
        'XLNX', 'ZM', 'ZS', 'ELF', 'MCHP', 'MRVL', 'CSGP', 'ANSS', 'EL',
    ]
    # Remover tickers problemáticos si los hay
    tickers = [ticker for ticker in tickers if ticker not in 
               ['MXIM', 'ATVI', 'CRM', 'REGN', 'SPLK', 'ZM', 'SWKS', 
                'SIRI', 'TSLA', 'SGEN', 'TXN']]
    return tickers

def fetch_data(ticker, start_date, end_date, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            print(f"\nDescargando datos para {ticker} desde {start_date.date()} hasta {end_date.date()} con intervalo '1h' (Intento {attempt + 1}/{retries})...")
            # Definir la ruta del archivo local
            filename = f"{ticker}.csv"
            if os.path.exists(filename):
                # Cargar datos localmente si existen
                data = pd.read_csv(filename, index_col=0, parse_dates=True)
                print(f"Datos cargados localmente para {ticker}.")
            else:
                # Descargar datos si no existen localmente
                data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False, auto_adjust=False)
                if data.empty:
                    print(f"No se encontraron datos para {ticker}.")
                    return None
                data.to_csv(filename)  # Guardar localmente
                print(f"Datos descargados y guardados localmente para {ticker}.")

            # Eliminar columnas duplicadas
            data = data.loc[:, ~data.columns.duplicated()]
            
            # Manejar 'Adj Close' y 'Close'
            if 'Close' in data.columns and 'Adj Close' in data.columns:
                data.drop(columns=['Adj Close'], inplace=True)
                print(f"Ticker: {ticker} | 'Adj Close' eliminado, manteniendo 'Close'.")
            elif 'Adj Close' in data.columns:
                data.rename(columns={'Adj Close': 'Close'}, inplace=True)
                print(f"Ticker: {ticker} | 'Adj Close' renombrado a 'Close'.")
            elif 'Close' not in data.columns:
                print(f"Error: 'Close' ni 'Adj Close' están presentes para {ticker}")
                return None

            # Seleccionar solo las columnas necesarias
            required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            existing_columns = [col for col in required_columns if col in data.columns]
            data = data[existing_columns]
            
            # Eliminar filas duplicadas de cabecera
            if data['Close'].dtype == object:
                data = data[data['Close'] != 'Close']
                print(f"Ticker: {ticker} | Filas duplicadas de cabecera eliminadas.")

            # Asegurar que 'Close' sea numérico
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            data = data.dropna(subset=['Close'])
            
            # Verificar si hay filas vacías después de las limpiezas
            if data.empty:
                print(f"Error: No hay datos válidos para {ticker} después de la limpieza.")
                return None

            # Eliminar cualquier columna duplicada residual
            data = data.loc[:, ~data.columns.duplicated()]
            
            # Verificar si 'Close' está duplicado
            close_cols = [col for col in data.columns if col == 'Close']
            if len(close_cols) > 1:
                # Mantener solo la primera columna 'Close'
                data = data.rename(columns={close_cols[0]: 'Close'})
                for col in close_cols[1:]:
                    data.drop(columns=[col], inplace=True)
                print(f"Ticker: {ticker} | Columnas 'Close' duplicadas eliminadas.")

            # Verificar la cantidad de datos después de la limpieza
            if len(data) < 100:  # Por ejemplo, al menos 100 puntos de datos
                print(f"Ticker: {ticker} | Datos insuficientes ({len(data)} puntos).")
                return None

            # Imprimir las columnas descargadas para depuración
            print(f"Ticker: {ticker} | Columnas: {list(data.columns)} | Puntos de Datos: {len(data)}")
            
            return data
        except yf.YFDownloadError as e:
            # Manejar errores específicos de yfinance
            attempt += 1
            print(f"Error al descargar datos para {ticker} (Intento {attempt}/{retries}): {e}")
            time.sleep(delay)  # Esperar antes del siguiente intento
        except Exception as e:
            # Manejar otros errores
            attempt += 1
            print(f"Error inesperado al descargar datos para {ticker} (Intento {attempt}/{retries}): {e}")
            time.sleep(delay)
    print(f"Fallo definitivo al descargar datos para {ticker}")
    return None

def prepare_data(tickers, start_date, end_date):
    all_data = []
    failed_tickers = []

    # Descargar datos en paralelo
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_data, ticker, start_date, end_date): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None and 'Close' in data.columns:
                all_data.append( (ticker, data) )
            else:
                failed_tickers.append(ticker)

    # Reportar tickers que fallaron
    if failed_tickers:
        print("\nTickers que no se pudieron descargar o no tienen datos disponibles:")
        for ft in failed_tickers:
            print(f"- {ft}")

    if not all_data:
        print("No se pudo descargar datos para ningún ticker.")
        return [], None, None

    return all_data, start_date, end_date

def backtest(all_data, start_ma, end_ma):
    best_results = {}

    # Iterar sobre cada ticker y realizar backtesting individual
    for ticker, data in all_data:
        # Verificar que 'Close' está presente
        if 'Close' not in data.columns:
            print(f"Error: 'Close' no está presente en los datos de {ticker}")
            continue

        close_series = data['Close']

        # Asegurarse de que 'close_series' es una Serie y no está vacía
        if not isinstance(close_series, pd.Series) or close_series.empty:
            print(f"Error: 'Close' no es una Serie válida para {ticker}")
            continue

        try:
            close_series = close_series.astype(float).copy()
        except ValueError as ve:
            print(f"Error al convertir 'Close' a float para {ticker}: {ve}")
            continue

        print(f"\nIniciando backtest para {ticker} con {len(close_series)} puntos de datos.")

        best_profit = float('-inf')
        best_fast_ma = None
        best_slow_ma = None

        # Definir el rango de medias móviles
        max_ma = min(end_ma, len(close_series))  # Ajustar el MA máximo según los datos disponibles

        for fast_ma in range(start_ma, max_ma + 1):
            for slow_ma in range(fast_ma + 1, max_ma + 1):
                try:
                    # Verificar que el window no excede la longitud de la serie
                    if slow_ma > len(close_series):
                        break  # No hay suficientes datos para esta combinación

                    # Calcular las medias móviles
                    fast_ma_series = close_series.rolling(window=fast_ma).mean()
                    slow_ma_series = close_series.rolling(window=slow_ma).mean()

                    # Crear un DataFrame con las medias móviles
                    df = pd.concat([close_series, fast_ma_series, slow_ma_series], axis=1)

                    # Verificar que la concatenación resultó en exactamente 3 columnas
                    if df.shape[1] != 3:
                        print(f"Advertencia: El DataFrame concatenado tiene {df.shape[1]} columnas en lugar de 3. Ticker: {ticker}")
                        continue

                    df.columns = ['Close', 'Fast_MA', 'Slow_MA']
                    df.dropna(inplace=True)

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
                    total_profit = profits_list.sum()

                    # Verificar y actualizar el mejor rendimiento
                    if total_profit > best_profit:
                        best_profit = total_profit
                        best_fast_ma = fast_ma
                        best_slow_ma = slow_ma
                except Exception as e:
                    print(f"Error al procesar MA para {ticker} (Fast MA: {fast_ma}, Slow MA: {slow_ma}): {e}")
                    continue

        if best_fast_ma and best_slow_ma:
            best_results[ticker] = {
                'Fast_MA': best_fast_ma,
                'Slow_MA': best_slow_ma,
                'Profit': best_profit
            }
            print(f"Ticker: {ticker} | Best Fast MA: {best_fast_ma} | Best Slow MA: {best_slow_ma} | Profit: {best_profit:.2f}")
        else:
            print(f"Ticker: {ticker} | No se encontraron combinaciones de medias móviles válidas.")

    return best_results

def main():
    tickers = get_nasdaq_100_tickers()
    
    # Definir fechas fijas para asegurar consistencia
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=60)  # 60 días
    
    all_data, _, _ = prepare_data(tickers, start_date, end_date)

    if not all_data:
        print("No hay datos suficientes para realizar el backtesting.")
        return

    print(f"\nDatos descargados y preparados. No se requieren fechas alineadas.\n")

    # Realizar el backtesting para cada ticker de manera independiente
    best_results = backtest(all_data, 1, 200)  # Rango reducido a 1-200

    # Resumen de los mejores resultados
    if best_results:
        print("\nResumen de las mejores combinaciones de medias móviles por ticker:")
        summary_df = pd.DataFrame(best_results).T
        print(summary_df)

        # Calcular estadísticas adicionales si lo deseas
        average_profit = summary_df['Profit'].mean()
        print(f"\nBeneficio promedio: {average_profit:.2f}")

        # Opcional: Guardar los resultados en un archivo CSV
        summary_df.to_csv('best_moving_averages.csv', index=True)
        print("\nLos resultados se han guardado en 'best_moving_averages.csv'.")
    else:
        print("No se encontraron resultados válidos para ningún ticker.")

if __name__ == "__main__":
    main()