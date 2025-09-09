import pandas as pd

def get_sp500_tickers():
    """
    Gets a list of S&P500 ticker symbols from Wikipedia for the user to select
    
    Returns:
    - List of S&P500 ticker symbols
    """

    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].tolist()
    return tickers