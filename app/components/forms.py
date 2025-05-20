# app/components/forms.py

import streamlit as st
from datetime import datetime
from typing import Optional

from typing import Optional
from datetime import datetime
import streamlit as st

def expense_form() -> Optional[dict]:
    """
    Form to add a new budget entry.

    Returns:
        dict or None: Data entered in the form or None if the form was not submitted.
    """

    with st.form("expense_form", clear_on_submit=True):
        date = st.date_input("Transaction Date", value=datetime.today())
        description = st.text_input("Description")
        category = st.selectbox("Category", [
            "Groceries", "Entertainment", "Transport", "Healthcare", 
            "Utilities", "Other", "Salary", "Gifts"
        ])
        type_ = st.selectbox("Transaction Type", ["Expense", "Income"])
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        payment_method = st.selectbox("Payment Method", ["Card", "Cash", "Bank Transfer"])
        status = st.selectbox("Status", ["Completed", "Pending"])

        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            return {
                "date": date,
                "description": description,
                "category": category,
                "type": type_,
                "amount": amount,
                "payment_method": payment_method,
                "status": status,
                "created_at": datetime.now()
            }
    return None

def investment_form() -> Optional[dict]:
    """
    Form to add a new investment.

    Returns:
        dict or None: Data entered in the form or None if the form was not submitted.
    """

    st.subheader("Add a New Investment")

    with st.form("investment_form", clear_on_submit=True):
        purchase_date = st.date_input("Purchase Date", value=datetime.today())
        company_name = st.text_input("Company Name")
        ticker_symbol = st.text_input("Ticker (e.g. AAPL, MSFT, TSLA)").upper()
        shares = st.number_input("Number of Shares", min_value=1, step=1)
        purchase_price_usd = st.number_input("Purchase Price (USD)", min_value=0.0, format="%.2f")
        account_type = st.selectbox("Account Type", ["Standard", "IKE", "IKZE"])
        status = st.selectbox("Investment Status", ["Active", "Sold"])

        submitted = st.form_submit_button("Add Investment")

        if submitted:
            return {
                "purchase_date": purchase_date,
                "company_name": company_name,
                "ticker_symbol": ticker_symbol,
                "shares": shares,
                "purchase_price_usd": purchase_price_usd,
                "account_type": account_type,
                "currency": "USD",  # Currently hardcoded to USD
                "status": status,
                "created_at": datetime.now()
            }
    return None
