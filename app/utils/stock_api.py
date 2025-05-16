# app/utils/stock_api.py
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
ALPHA_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

def get_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Pobiera dane giełdowe dla danego tickera z yfinance, a jeśli się nie uda – z Alpha Vantage.
    """
    # 1. Próba z yfinance
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            return df
        else:
            print(f"$ {ticker}: yfinance returned empty data (period={period})")
    except Exception as e:
        print(f"$ {ticker}: yfinance error: {e}")

    # 2. Fallback do Alpha Vantage
    try:
        print(f"$ {ticker}: Falling back to Alpha Vantage...")
        ts = TimeSeries(key=ALPHA_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='compact')

        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)

        data.reset_index(inplace=True)
        data.rename(columns={'date': 'Date'}, inplace=True)

        return data
    except Exception as e:
        print(f"$ {ticker}: Alpha Vantage failed: {e}")
        return None

