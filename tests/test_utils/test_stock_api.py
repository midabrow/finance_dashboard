# tests/test_utils/test_stock_api.py

from app.utils.stock_api import format_statement, get_key_metrics, evaluate_company
from plotly.graph_objects import Figure
import plotly.express as px
from app.utils.stock_api import plot_price_chart_candles, plot_moving_averages, plot_income
import pandas as pd


def test_format_statement_with_fixture(sample_income_df):
    # Given
    df = sample_income_df

    # When
    result = format_statement(df)

    # Then
    assert result.shape == (2, 3)
    assert list(result.index) == ['2022-12-31', '2023-12-31']
    assert list(result.columns) == ['Total Revenue', 'Gross Profit', 'Net Income']

def test_get_key_metrics_returns_expected_fields(mocker):

    # Tworzymy sztucznego Ticker'a
    fake_info = {
        "currentPrice": 100.5,
        "previousClose": 99.8,
        "marketCap": 1500000000,
        "trailingPE": 12.3,
        "forwardPE": 11.7,
        "trailingEps": 5.6,
        "dividendYield": 0.02,
        "returnOnEquity": 0.18,
        "returnOnAssets": 0.11,
        "debtToEquity": 0.7,
        "priceToBook": 2.3
    }

    mock_ticker = mocker.patch("app.utils.stock_api.yf.Ticker")
    mock_ticker.return_value.info = fake_info

    result = get_key_metrics("AAPL")

    assert isinstance(result, dict)
    assert result["Current Price"] == 100.5
    assert result["P/E (TTM)"] == 12.3
    assert result["Dividend Yield"] == 0.02


def test_plot_price_chart_candles_valid_data(mocker):
    df = pd.DataFrame({
        "Open": [100, 105],
        "High": [110, 115],
        "Low": [90, 95],
        "Close": [105, 100]
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))


    mocker.patch("app.utils.stock_api.yf.download", return_value=df)
    mocker.patch("app.utils.stock_api.st.error") # nie wyświetlaj errorów

    fig = plot_price_chart_candles("AAPL")
    assert isinstance(fig, Figure)
    assert len(fig.data) > 0
    assert fig.layout.title.text.startswith("AAPL")

def test_plot_price_chart_candles_empty(mocker):
    # Zwracamy pusty DataFrame
    mocker.patch("app.utils.stock_api.yf.download", return_value=pd.DataFrame())
    mock_error = mocker.patch("app.utils.stock_api.st.error")

    fig = plot_price_chart_candles("AAPL")
    assert isinstance(fig, Figure)
    assert fig.data == ()
    mock_error.assert_called_once()

def test_plot_moving_averages_valid_data(mocker):
    df = pd.DataFrame({
        "Close": [i for i in range(100)]
    }, index=pd.date_range("2022-01-01", periods=100))

    mocker.patch("app.utils.stock_api.yf.download", return_value=df)
    mocker.patch("app.utils.stock_api.st.error")

    fig = plot_moving_averages("AAPL")
    assert isinstance(fig, px.line().__class__)
    assert fig.data
    assert fig.layout.title.text.startswith("AAPL")

def test_plot_moving_averages_empty(mocker):
    mocker.patch("app.utils.stock_api.yf.download", return_value=pd.DataFrame())
    mock_error = mocker.patch("app.utils.stock_api.st.error")

    fig = plot_moving_averages("AAPL")
    assert fig.data == ()
    mock_error.assert_called_once()

def test_plot_income_valid_df(sample_income_df):
    fig = plot_income(sample_income_df.T)
    assert fig.data # coś zostało narysowane

def test_plot_income_missing_column(sample_income_df, mocker):

    df = sample_income_df.T.drop(columns="Total Revenue")
    mocker_error = mocker.patch("app.utils.stock_api.st.error")
    
    fig = plot_income(df)
    assert fig.data == ()
    mocker_error.asser_called_once()

def test_evaluate_company_typical():
    metrics = {
        "P/E (TTM)": 14,
        "ROE": 0.2,
        "Debt to Equity": 0.8
    }

    result = evaluate_company(metrics)
    assert "P/E" in result[0]
    assert "ROE" in result[1]
    assert "Debt" in result[2]   

def test_evaluate_company_high_pe():
    metrics = {
        "P/E (TTM)": 50,
        "ROE": 0.1,
        "Debt to Equity": 2.0
    }
    result = evaluate_company(metrics)
    assert any("P/E" in line for line in result)