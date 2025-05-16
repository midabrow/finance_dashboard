import streamlit as st

from sections import home, investments, stock_tracker, predictions

st.set_page_config(
    page_title = "Finance Dashboard",
    page_icon = "💸",
    layout = "wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to:",
    ("Home Budget", "Investment Portfolio", "Stock Tracker", "Price Forecasting")
)

if page == "Home Budget":
    home.show_home_page()
elif page == "Investment Portfolio":
    investments.show_investments_page()
elif page == "Stock Tracker":
    stock_tracker.show_stock_tracker_page()
elif page == "Price Forecasting":
    predictions.show_predictions_page()
else:
    st.error("Page not found!")
