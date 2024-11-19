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

        # Opcional: Mostrar progreso cada cierto número de iteraciones
        if fast_ma % 20 == 0:
            print(f"Progreso: Fast MA = {fast_ma}/{max_ma}")

    return best_fast_ma, best_slow_ma, best_profit

def main():
    tickers = get_nasdaq_100_tickers()
    all_data, start_date, end_date = prepare_data(tickers)

    if not all_data:
        print("No hay datos suficientes para realizar el backtesting.")
        return

    print(f"\nDatos descargados y preparados. No se requieren fechas alineadas.\n")

    # Considera reducir el rango para pruebas iniciales
    best_fast_ma, best_slow_ma, best_profit = backtest(all_data, 1, 1000)

    print(f"\nLas mejores medias móviles son:")
    print(f"Fast MA: {best_fast_ma}")
    print(f"Slow MA: {best_slow_ma}")
    print(f"Beneficio Total: {best_profit:.2f}")

if __name__ == "__main__":
    main()