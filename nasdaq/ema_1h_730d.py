import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count

def fetch_data(ticker, period="max", interval="1h"):
    """Función para descargar datos históricos de Yahoo Finance."""
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No se encontraron datos.")
        return data
    except Exception as e:
        print(f"Error al descargar datos para {ticker}: {e}")
        return None

def moving_average_strategy(data, fast_period, slow_period):
    """Función que implementa la estrategia de cruce de medias móviles exponenciales."""
    data[f'EMA_{fast_period}'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
    data[f'EMA_{slow_period}'] = data['Close'].ewm(span=slow_period, adjust=False).mean()

    # Genera señales y convierte a arrays de NumPy para evitar ambigüedades
    signal = np.where(data[f'EMA_{fast_period}'] > data[f'EMA_{slow_period}'], 1, 0)
    position = np.diff(signal, prepend=0)

    profit = 0.0  # Asegúrate de que profit sea un valor numérico
    entry_price = None

    for i in range(len(data)):
        if position[i] == 1:  # Señal de compra
            entry_price = data['Close'].iloc[i]
        elif position[i] == -1 and entry_price is not None:  # Señal de venta
            profit += data['Close'].iloc[i] - entry_price
            entry_price = None

    # Aseguramos que profit es un valor numérico único y no una Serie
    return profit if isinstance(profit, (float, int)) else float(profit.iloc[0] if isinstance(profit, pd.Series) else profit)

def backtest_company(ticker):
    """Backtesting para una empresa con todas las combinaciones de medias móviles."""
    data = fetch_data(ticker)
    if data is None:
        return (ticker, (0, 0), 0.0)

    max_profit = -np.inf
    best_pair = (0, 0)
    count = 0  # contador para imprimir progreso

    for fast, slow in product(range(1, 101), repeat=2):
        if fast < slow:
            profit = moving_average_strategy(data.copy(), fast, slow)
            if profit > max_profit:
                max_profit = profit
                best_pair = (fast, slow)
            
            # Incrementa el contador y muestra el progreso cada 100 combinaciones
            count += 1
            if count % 10 == 0:  # Cambia 100 según prefieras
                print(f"Ticker: {ticker}, Fast MA: {fast}, Slow MA: {slow}, Current Profit: {profit}")

    return ticker, best_pair, max_profit

def analyze_ticker(args):
    """Función de ayuda para procesar cada ticker en paralelo."""
    return backtest_company(args)

def backtest_nasdaq100():
    """Backtesting para todas las empresas del NASDAQ-100 usando paralelización."""
    nasdaq100 = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'META', 'TSLA', 'PEP', 'AZN', 
        'LIN', 'CSCO', 'ADBE', 'QCOM', 'TXN', 'ISRG', 'INTU', 'AMGN', 'PDD', 'CMCSA',
        'AMAT', 'ARM', 'BKNG', 'HON', 'VRTX', 'MU', 'PANW', 'ADP', 'ADI', 'GILD',
        'SBUX', 'MELI', 'REGN', 'LRCX', 'INTC', 'MDLZ', 'KLAC', 'ABNB', 'CTAS',
        'PYPL', 'CEG', 'SNPS', 'MAR', 'CRWD', 'MRVL', 'CDNS', 'ORLY', 'CSX', 'DASH',
        'WDAY', 'NXPI', 'ADSK', 'FTNT', 'TTD', 'ROP', 'PCAR', 'FANG', 'MNST', 'AEP',
        'PAYX', 'CPRT', 'TEAM', 'CHTR', 'ROST', 'KDP', 'FAST', 'DDOG', 'ODFL', 'KHC',
        'MCHP', 'GEHC', 'EXC', 'EA', 'VRSK', 'LULU'
    ]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(analyze_ticker, nasdaq100)
    
    # Organizar y mostrar resultados
    results = {ticker: {"Best Pair": best_pair, "Profit": profit} for ticker, best_pair, profit in results}
    best_common_pair = max(results, key=lambda x: results[x]["Profit"])

    for ticker, result in results.items():
        print(f"{ticker}: Mejor cruce {result['Best Pair']} con beneficio de {result['Profit']:.2f}")

    print(f"\nMejor cruce común en todas las empresas: {results[best_common_pair]['Best Pair']} con beneficio de {results[best_common_pair]['Profit']:.2f}")

# Ejecutar el análisis
if __name__ == "__main__":
    backtest_nasdaq100()