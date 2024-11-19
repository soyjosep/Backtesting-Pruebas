import yfinance as yf

def test_ticker(ticker):
    print(f"\nDescargando datos para {ticker} con intervalo '1h' y periodo de 730 días...")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=730)
    data = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False, auto_adjust=False)
    if data.empty:
        print(f"No se encontraron datos para {ticker}.")
        return
    print(f"Columnas descargadas para {ticker}: {list(data.columns)}")
    print(data.head())

if __name__ == "__main__":
    import datetime
    test_ticker('AAPL')  # Puedes cambiar 'AAPL' por otro ticker si lo deseas
    