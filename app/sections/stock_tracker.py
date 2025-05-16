# app/pages/stock_tracker.py

import streamlit as st
import plotly.express as px
from utils.stock_api import get_stock_data

def show_stock_tracker_page():
    """
    Page – Stock Price Tracker.
    """

    st.title("📊 Stock Price Tracker")
    st.markdown("---")

    st.subheader("🔎 Search for a Company")

    ticker = st.text_input("Enter stock ticker (e.g. AAPL, MSFT, TSLA)").upper()

    period_options = {
        "1 day": "1d",
        "5 days": "5d",
        "1 month": "1mo",
        "6 months": "6mo",
        "1 year": "1y",
        "5 years": "5y",
        "maximum": "max"
    }

    period_choice = st.selectbox("Time Range", options=list(period_options.keys()))

    if ticker:
        st.subheader(f"📈 Price Chart for: {ticker}")

        # Fetch historical stock data
        stock_data = get_stock_data(ticker, period=period_options[period_choice], interval="1d")

        if stock_data is not None:
            fig = px.line(
                stock_data,
                x="Date",
                y="Close",
                title=f"Closing Price for {ticker}",
                labels={"Date": "Date", "Close": "Closing Price (USD)"},
                markers=True
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Chart successfully loaded!")
        else:
            st.error(f"❌ No data available for ticker '{ticker}'. Please check if the symbol is correct.")
    else:
        st.info("Enter a stock ticker to display the chart.")
