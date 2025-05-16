# app/pages/predictions.py

import streamlit as st
import plotly.express as px
from utils.stock_api import get_stock_data
from prophet import Prophet
import pandas as pd

def show_predictions_page():
    """
    Page – Stock Price Prediction.
    """

    st.title("🔮 Stock Price Forecast")
    st.markdown("---")

    st.subheader("🔎 Select a Company to Forecast")

    ticker = st.text_input("Enter stock ticker (e.g. AAPL, MSFT, TSLA)").upper()

    prediction_period = st.selectbox("Forecast Horizon", ["30 days", "90 days", "180 days"])

    if ticker:
        st.subheader(f"📈 Historical Data: {ticker}")

        # Load historical stock data (1 year)
        stock_data = get_stock_data(ticker, period="1y", interval="1d")

        if stock_data is not None and not stock_data.empty:
            df = stock_data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

            st.line_chart(df.set_index("ds")["y"])

            st.subheader("📈 Modeling and Prediction")

            # Create and train Prophet model
            model = Prophet()
            model.fit(df)

            # Generate future dates
            horizon_days = int(prediction_period.split()[0])
            future = model.make_future_dataframe(periods=horizon_days)

            # Generate forecast
            forecast = model.predict(future)

            # Display forecast chart
            fig = px.line(
                forecast,
                x="ds",
                y="yhat",
                labels={"ds": "Date", "yhat": "Predicted Price (USD)"},
                title=f"{ticker} Stock Price Forecast for {horizon_days} Days",
                markers=True
            )

            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Forecast successfully generated!")
        else:
            st.error(f"❌ No data found for ticker '{ticker}'. Please check if the symbol is correct.")
    else:
        st.info("Enter a stock ticker to begin forecasting.")
