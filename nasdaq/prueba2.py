import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time

# Ignorar advertencias de pandas para una salida más limpia
warnings.filterwarnings("ignore")

def get_nasdaq_100_tickers():
    tickers = [
        'ATVI',   # Activision Blizzard
        'ADBE',   # Adobe Inc.
        'AMD',    # Advanced Micro Devices
        'A',      # Agilent Technologies
        'ABNB',   # Airbnb
        'ALGN',   # Align Technology
        'ALLE',   # Allegion
        'GOOGL',  # Alphabet Inc. Class A
        'GOOG',   # Alphabet Inc. Class C
        'AMZN',   # Amazon.com
        'AMGN',   # Amgen Inc.
        'ADI',    # Analog Devices
        'AAPL',   # Apple Inc.
        'AMAT',   # Applied Materials
        'ASML',   # ASML Holding
        'ADSK',   # Autodesk
        'ADP',    # Automatic Data Processing
        'BIDU',   # Baidu Inc.
        'BIIB',   # Biogen Inc.
        'BMRN',   # BioMarin Pharmaceutical
        'BKNG',   # Booking Holdings
        'AVGO',   # Broadcom Inc.
        'CDNS',   # Cadence Design Systems
        'CDW',    # CDW Corporation
        'CHTR',   # Charter Communications
        'CHKP',   # Check Point Software Technologies
        'CTAS',   # Cintas Corporation
        'CSCO',   # Cisco Systems
        'CTSH',   # Cognizant Technology Solutions
        'CMCSA',  # Comcast Corporation
        'COP',    # ConocoPhillips
        'CPRT',   # Copart Inc.
        'COST',   # Costco Wholesale Corporation
        'CRWD',   # CrowdStrike Holdings
        'DOCU',   # DocuSign
        'DLTR',   # Dollar Tree
        'EBAY',   # eBay Inc.
        'EA',     # Electronic Arts
        'EXC',    # Exelon Corporation
        'EXPE',   # Expedia Group
        'FAST',   # Fastenal
        'META',   # Meta Platforms Inc. (Facebook)
        'FISV',   # Fiserv Inc.
        'FOXA',   # Fox Corporation Class A
        'FOX',    # Fox Corporation Class B
        'GILD',   # Gilead Sciences
        'HAS',    # Hasbro
        'HSIC',   # Henry Schein
        'HLT',    # Hilton Worldwide
        'HMC',    # Honda Motor Co. Ltd.
        'INTU',   # Intuit
        'JD',     # JD.com
        'KLAC',   # KLA Corporation
        'KHC',    # Kraft Heinz Company
        'LRCX',   # Lam Research
        'LBTYA',  # Liberty Global Class A
        'LBTYK',  # Liberty Global Class K
        'MAR',    # Marriott International
        'MELI',   # MercadoLibre
        'MSFT',   # Microsoft Corporation
        'MRNA',   # Moderna
        'MDLZ',   # Mondelez International
        'NTES',   # NetEase
        'NFLX',   # Netflix
        'NVDA',   # NVIDIA Corporation
        'ORLY',   # O'Reilly Automotive
        'PAYX',   # Paychex
        'PYPL',   # PayPal Holdings
        'PEP',    # PepsiCo
        'QCOM',   # QUALCOMM
        'REGN',   # Regeneron Pharmaceuticals
        'ROST',   # Ross Stores
        'SPGI',   # S&P Global
        'CRM',    # Salesforce
        'SGEN',   # Seagen Inc.
        'SIRI',   # Sirius XM Holdings
        'SWKS',   # Skyworks Solutions
        'SPLK',   # Splunk
        'SBUX',   # Starbucks
        'TTWO',   # Take-Two Interactive
        'TMUS',   # T-Mobile US
        'TSLA',   # Tesla
        'TXN',    # Texas Instruments
        'ISRG',   # Intuitive Surgical
        'VRSN',   # VeriSign
        'VRTX',   # Vertex Pharmaceuticals
        'WBA',    # Walgreens Boots Alliance
        'WDC',    # Western Digital
        'WDAY',   # Workday
        'XLNX',   # Xilinx
        'ZM',     # Zoom Video Communications
        'ZS',     # Zscaler
        'ELF',    # e.l.f. Beauty
        'MCHP',   # Microchip Technology
        'MRVL',   # Marvell Technology
        'CSGP',   # CoStar Group
        'ANSS',   # ANSYS
        'EL',     # Estée Lauder Companies Inc.
    ]
    # Removemos 'MXIM' ya que ha sido delistado
    tickers = [ticker for ticker in tickers if ticker != 'MXIM']
    return tickers

def fetch_data(ticker, start_date, end_date, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
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
            attempt += 1
            print(f"Error al descargar datos para {ticker} (Intento {attempt}/{retries}): {e}")
            time.sleep(delay)  # Esperar antes del siguiente intento
    print(f"Fallo definitivo al descargar datos para {ticker}")
    return None

def prepare_data(tickers):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730)  # Últimos 730 días

    all_data = []
    failed_tickers = []

    # Descargar datos en paralelo
    with ThreadPoolExecutor(max_workers=20) as executor:  # Incrementamos workers para acelerar
        futures = {executor.submit(fetch_data, ticker, start_date, end_date): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            data = future.result()
            if data is not None and not data.empty:
                all_data.append(data)
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
    for data in all_data:
        ticker = data['Ticker'].iloc[0]
        close_series = data['Close'].astype(float).copy()

        best_profit = float('-inf')
        best_fast_ma = None
        best_slow_ma = None

        # Definir el rango de medias móviles
        max_ma = min(100, end_ma)  # Ajusta este valor según tus necesidades

        for fast_ma in range(start_ma, max_ma + 1):
            for slow_ma in range(fast_ma + 1, max_ma + 1):
                # Calcular las medias móviles
                fast_ma_series = close_series.rolling(window=fast_ma).mean()
                slow_ma_series = close_series.rolling(window=slow_ma).mean()

                # Crear un DataFrame con las medias móviles
                df = pd.concat([close_series, fast_ma_series, slow_ma_series], axis=1)
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
    all_data, start_date, end_date = prepare_data(tickers)

    if not all_data:
        print("No hay datos suficientes para realizar el backtesting.")
        return

    print(f"\nDatos descargados y preparados. No se requieren fechas alineadas.\n")

    # Realizar el backtesting para cada ticker de manera independiente
    best_results = backtest(all_data, 1, 1000)

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