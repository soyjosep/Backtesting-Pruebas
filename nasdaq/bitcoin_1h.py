import pandas as pd
import yfinance as yf
from tqdm import tqdm
import ta
from multiprocessing import Pool, cpu_count

# Parámetros del backtesting
start_date = "2022-11-01"  # Fecha de creación de Bitcoin
end_date = "2024-10-28"  # Fecha actual o última disponible

# Descargar datos históricos de Bitcoin
data = yf.download("BTC-USD", start=start_date, end=end_date, interval="1h")

# Verificar si los datos fueron descargados correctamente
if data.empty:
    print("Error: No se pudieron descargar los datos históricos. Intenta con un rango de fechas diferente.")
else:
    # Función para ejecutar el backtesting
    def run_backtest(slow_period, fast_period, data):
        df = data.copy()
        try:
            close_series = df['Close'].squeeze()

            # Calcular las medias móviles
            df['Slow_MA'] = ta.trend.SMAIndicator(close_series, slow_period).sma_indicator()
            df['Fast_MA'] = ta.trend.SMAIndicator(close_series, fast_period).sma_indicator()
        except Exception as e:
            print(f"Error al calcular SMA con períodos Slow={slow_period} y Fast={fast_period}: {e}")
            return float("-inf")

        # Variables de backtesting
        position = 0  # 0 = sin posición, 1 = posición de compra
        buy_price = 0
        profit = 0.0  # Asegurar que profit sea un float

        for i in range(1, len(df)):
            if pd.notna(df['Fast_MA'].iloc[i]) and pd.notna(df['Slow_MA'].iloc[i]):
                # Condición de compra: cruce al alza de medias móviles
                if position == 0 and df['Fast_MA'].iloc[i] > df['Slow_MA'].iloc[i] and df['Fast_MA'].iloc[i-1] <= df['Slow_MA'].iloc[i-1]:
                    position = 1
                    buy_price = df['Close'].iloc[i].item()  # Convertir a float con .item()
                # Condición de cierre de posición: cruce a la baja de medias móviles
                elif position == 1 and df['Fast_MA'].iloc[i] < df['Slow_MA'].iloc[i] and df['Fast_MA'].iloc[i-1] >= df['Slow_MA'].iloc[i-1]:
                    position = 0
                    profit += df['Close'].iloc[i].item() - buy_price  # Convertir a float con .item()

        # Cerrar la última posición abierta
        if position == 1:
            profit += df['Close'].iloc[-1].item() - buy_price

        return profit

    # Mover process_combination fuera de find_best_ma_combination
    def process_combination(params):
        slow_ma, fast_ma, data = params
        profit = run_backtest(slow_ma, fast_ma, data)
        return slow_ma, fast_ma, profit

    # Función para encontrar la mejor combinación de medias móviles
    def find_best_ma_combination(data):
        best_slow_ma = None
        best_fast_ma = None
        best_profit = float("-inf")

        # Generar todas las combinaciones de medias móviles posibles
        combinations = [(slow_ma, fast_ma, data) for slow_ma in range(1, 51) for fast_ma in range(1, slow_ma)]
        
        with Pool(processes=cpu_count()) as pool:
            for slow_ma, fast_ma, profit in tqdm(pool.imap_unordered(process_combination, combinations), total=len(combinations), desc="Backtesting Combinations"):
                if profit > best_profit:
                    best_profit = profit
                    best_slow_ma = slow_ma
                    best_fast_ma = fast_ma
                    print(f"Mejor combinación temporal: Lenta={best_slow_ma}, Rápida={best_fast_ma}, Profit={best_profit}")

        return best_slow_ma, best_fast_ma, best_profit

    # Bloque principal
    if __name__ == '__main__':
        # Encontrar la mejor combinación
        best_slow_ma, best_fast_ma, best_profit = find_best_ma_combination(data)

        # Mostrar el mejor resultado
        print("\nMejor combinación de medias móviles:")
        print(f"Media Móvil Lenta: {best_slow_ma}, Media Móvil Rápida: {best_fast_ma}")
        print(f"Profit obtenido: {best_profit}")