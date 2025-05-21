# app/services/investment_actions.py

from models.investment import Investment
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
import yfinance as yf
from typing import Optional
import pandas as pd 

def add_stocks(db: Session, form_data: dict) -> None:
    """
    Saves a new investment transaction to the database.
    """
    investment = Investment(
        purchase_date=form_data["purchase_date"],
        company_name=form_data["company_name"],
        ticker_symbol=form_data["ticker_symbol"],
        shares=form_data["shares"],
        purchase_price_usd=form_data["purchase_price_usd"],
        account_type=form_data["account_type"],
        currency=form_data["currency"],
        status=form_data["status"],
        created_at=datetime.now()
    )
    db.add(investment)
    db.commit()


def get_wallet(db: Session) -> pd.DataFrame:
    """
    Returns current portfolio summary from the database.

    Args:
        db (Session): SQLAlchemy session

    Returns:
        pd.DataFrame: Grouped wallet with Shares and Investment Value
    """
    active = db.query(Investment).filter(Investment.status == "Active").all()

    df = pd.DataFrame([{
        "Company": inv.company_name,
        "Ticker": inv.ticker_symbol,
        "Shares": inv.shares,
        "Investment Value": inv.shares * float(inv.purchase_price_usd)
    } for inv in active])

    if df.empty:
        return df
    
    return (
        df.groupby(['Ticker', 'Company'], as_index=False).agg({"Shares": "sum", "Investment Value": "sum"})
    )


def get_active_investments_by_company(db: Session, company_name: str) -> List[Investment]:
    """
    Returns all active investments for a given company, sorted by purchase date.
    """
    return db.query(Investment).filter(
        Investment.status == "Active",
        Investment.company_name == company_name
    ).order_by(Investment.purchase_date.asc()).all()


def sell_stocks(db: Session, company_name: str, shares_to_sell: int) -> Optional[float]:
    """
    Processes sale of stock for a specific company by FIFO logic.

    Args:
        db (Session): SQLAlchemy session.
        company_name (str): Name of the company whose shares are to be sold.
        shares_to_sell (int): Number of shares to sell.

    Returns:
        float: Sell price per share (current market price).
    """
    remaining = shares_to_sell
    current_price = None

    active_investments = get_active_investments_by_company(db, company_name)

    if not active_investments:
        raise ValueError(f"No active investments found for company: {company_name}")
    
    ticker_symbol = active_investments[0].ticker_symbol
    price_data = yf.Ticker(ticker_symbol).history(period="1d")
    if price_data.empty:
        raise ValueError("Cannot fetch current market price.")

    market_price = price_data["Close"].iloc[-1]
    remaining = shares_to_sell

    for inv in active_investments:
        if remaining <= 0:
            break
        sell_now = min(inv.shares, remaining)

        #1. Reduce the amount of stock in the original listing
        inv.shares -= shares_to_sell
        if inv.shares == 0:
            inv.status == "Sold"
        
        #2. Add a new entry with this sale
        sold_investment = Investment(
            purchase_date = inv.purchase_date,
            company_name = inv.company_name,
            ticker_symbol = inv.ticker_symbol,
            shares = sell_now,
            purchase_price_usd = inv.purchase_price_usd,
            account_type = inv.account_type,
            currency = inv.currency,
            status = "Sold",
            sell_price_usd = current_price,
            sold_date = datetime.today(),
            created_at = datetime.now()
        )
        db.add(sold_investment)
        remaining -= sell_now

    db.commit()
