# tests/test_utils/test_helpers.py

import pytest
from unittest.mock import patch
import pandas as pd
from app.utils.helpers import get_sp500_tickers

def test_get_sp500_tickers_returns_list():
    tickers = get_sp500_tickers()

    # Czy wynik to lista?
    assert isinstance(tickers, list)

    # Czy lista nie jest pusta?
    assert len(tickers) > 0

    # Czy lista zawiera znane tickery?
    assert "AAPL" in tickers or "MSFT" in tickers


@patch("app.utils.helpers.pd.read_html")
def test_get_sp500_tickers_with_mock(mock_read_html):
    # Sztuczne DataFrame
    fake_data = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOG"]})
    mock_read_html.return_value = [fake_data]

    tickers = get_sp500_tickers()

    assert tickers == ["AAPL", "MSFT", "GOOG"]


@patch("app.utils.helpers.pd.read_html")
def test_get_sp500_tickers_missing_column(mock_read_html):
    fake_data = pd.DataFrame({"NewColumn": ["AAPL", "MSFT", "GOOG"]})
    mock_read_html.return_value = [fake_data]

    with pytest.raises(KeyError):
        get_sp500_tickers()

@patch("app.utils.helpers.pd.read_html")
def test_get_sp500_tickers_empty(mock_read_html):
    fake_data = pd.DataFrame(columns=["Symbol"])
    mock_read_html.return_value = [fake_data]

    tickers = get_sp500_tickers()

    assert isinstance(tickers, list)
    assert tickers == []