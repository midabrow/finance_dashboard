# app/pages/investments.py

import streamlit as st
from sqlalchemy.orm import Session
from components.database import get_db
from models.investment import Investment
from components.forms import investment_form
from components.plots import plot_investment_value
import pandas as pd
from utils.styles import styled_text

def show_investments_page():
    """
    Page – Investment Portfolio.
    """

    st.title("📈 Investment Portfolio")
    st.markdown("---")

    # Create database session
    db_generator = get_db()
    db: Session = next(db_generator)

    # Add new investment
    st.markdown(styled_text("➕ Add New Investment", color="#10B981", font_size="18px", margin_bottom="1rem"), unsafe_allow_html=True)
    new_investment = investment_form()

    if new_investment:
        investment = Investment(**new_investment)
        db.add(investment)
        db.commit()
        st.success("✅ New investment added successfully!")

    st.markdown("---")

    # Display existing investments
    st.markdown(styled_text("📋 Your Investments", color="#3B82F6", font_size="18px", margin_bottom="1rem"), unsafe_allow_html=True)

    investments = db.query(Investment).filter(Investment.status == "Active").all()

    if investments:
        investments_data = [{
            "Purchase Date": inv.purchase_date,
            "Company": inv.company_name,
            "Ticker": inv.ticker_symbol,
            "Shares": inv.shares,
            "Purchase Price (USD)": inv.purchase_price_usd,
            "Account Type": inv.account_type,
            "Status": inv.status
        } for inv in investments]

        df_investments = pd.DataFrame(investments_data)

        st.dataframe(df_investments)

        # Basic stats
        total_shares = sum(inv.shares for inv in investments)
        total_investment_value = sum(inv.shares * float(inv.purchase_price_usd) for inv in investments)

        col1, col2 = st.columns(2)
        col1.metric("📈 Total Shares", total_shares)
        col2.metric("💵 Portfolio Value (USD)", f"{total_investment_value:,.2f}")

        # Plot by company
        st.markdown(styled_text("📊 Investment Value by Company", color="#F59E0B", font_size="16px", margin_top="1.5rem"), unsafe_allow_html=True)
        st.plotly_chart(plot_investment_value(df_investments), use_container_width=True)

    else:
        st.info("No active investments found. Add your first investment! 🚀")

    # Close DB session
    try:
        db_generator.close()
    except:
        pass
