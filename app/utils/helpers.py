import pandas as pd
# import numpy as np
# import yfinance as yf

# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.express as px
# from sklearn.preprocessing import MinMaxScaler 
# from keras.models import load_model # type: ignore

def get_sp500_tickers():
    """
    Gets a list of S&P500 ticker symbols from Wikipedia for the user to select
    
    Returns:
    - List of S&P500 ticker symbols
    """

    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].tolist()
    return tickers