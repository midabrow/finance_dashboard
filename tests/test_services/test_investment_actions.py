from app.services.investment_actions import (
    add_stocks, get_wallet, get_active_investments_by_company, sell_stocks
)
from app.models.investment import Investment
from datetime import datetime
import pandas as pd
import pytest

def test_add_stocks_creates_investment(db_session):
    form_data = {
        "purchase_date": "2024-03-01",
        "company_name": "Tesla Inc.",
        "ticker_symbol": "TSLA",
        "shares": 3,
        "purchase_price_usd": 800.00,
        "account_type": "Zwykłe",
        "currency": "USD",
        "status": "Active"
    }

    add_stocks(db_session, form_data)
    result = db_session.query(Investment).filter_by(ticker_symbol="TSLA").first()

    assert result is not None
    assert result.company_name == "Tesla Inc."
    assert result.status == "Active"
    assert result.shares == 3

def test_get_wallet_aggregates_by_company(db_session):
    # Dodaj 2 rekordy tego samego tickera
    i1 = Investment(
        purchase_date="2024-01-01",
        company_name="Apple Inc.",
        ticker_symbol="AAPL",
        shares=2,
        purchase_price_usd=100
    )
    i2 = Investment(
        purchase_date="2024-01-15",
        company_name="Apple Inc.",
        ticker_symbol="AAPL",
        shares=3,
        purchase_price_usd=120
    )
    db_session.add_all([i1, i2])
    db_session.commit()

    df = get_wallet(db_session)
    assert not df.empty
    assert "Investment Value" in df.columns
    assert df.loc[0, "Shares"] == 5

def test_get_active_investments_by_company_sorted(db_session):
    db_session.add_all([
        Investment(purchase_date="2024-01-10", company_name="NVIDIA", ticker_symbol="NVDA", shares=1, purchase_price_usd=500),
        Investment(purchase_date="2024-01-01", company_name="NVIDIA", ticker_symbol="NVDA", shares=1, purchase_price_usd=500),
    ])
    db_session.commit()

    results = get_active_investments_by_company(db_session, "NVIDIA")
    assert len(results) == 2
    assert results[0].purchase_date < results[1].purchase_date

@pytest.mark.slow
def test_sell_stocks_creates_sold_entry(db_session, mocker):
    # --- Arrange: dodaj aktywną inwestycję do bazy ---
    inv = Investment(
        purchase_date="2024-01-01",
        company_name="Apple Inc.",
        ticker_symbol="AAPL",
        shares=3,
        purchase_price_usd=150,
        status="Active"
    )
    db_session.add(inv)
    db_session.commit()

    # --- Mock yfinance ---
    mock_price_df = pd.DataFrame({"Close": [190.0]})
    mock_yf = mocker.patch("app.services.investment_actions.yf.Ticker")
    mock_yf.return_value.history.return_value = mock_price_df

    # --- Act: sprzedaj 2 akcje ---
    sell_stocks(db_session, company_name="Apple Inc.", shares_to_sell=2)

    # --- Assert: sprawdzamy dane ---
    active_after = db_session.query(Investment).filter_by(status="Active").first()
    sold = db_session.query(Investment).filter_by(status="Sold").first()

    assert active_after.shares == 1
    assert sold is not None
    assert sold.shares == 2
    assert float(sold.sell_price_usd) == 190.0


import pytest
from app.services.investment_actions import add_stocks
from app.models.investment import Investment

@pytest.mark.parametrize("ticker,shares", [
    ("AAPL", 5),
    ("NVDA", 3),
    ("TSLA", 7),
])
def test_add_stocks_multiple_tickers(db_session, ticker, shares):
    form_data = {
        "purchase_date": "2024-01-01",
        "company_name": "Test Inc.",
        "ticker_symbol": ticker,
        "shares": shares,
        "purchase_price_usd": 100.00,
        "account_type": "Zwykłe",
        "currency": "USD",
        "status": "Active"
    }

    add_stocks(db_session, form_data)
    result = db_session.query(Investment).filter_by(ticker_symbol=ticker).first()
    assert result is not None
    assert result.shares == shares
