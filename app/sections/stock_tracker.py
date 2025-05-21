# app/pages/stock_tracker.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.stock_api import plot_price_chart_candles, plot_moving_averages, get_key_metrics, format_statement, plot_income, evaluate_company
from utils.helpers import get_sp500_tickers
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import datetime as dt
from components.database import get_db
from models.investment import Investment
from components.forms import investment_form
from components.plots import plot_investment_value
import pandas as pd
from utils.styles import styled_text
from sqlalchemy.orm import Session
from services.investment_actions import sell_stocks, add_stocks, get_wallet


def show_stock_tracker_page() -> None:
    """
    Main entry point for the stock tracker page.
    """
    st.title("📊 Stock Price Tracker")
    db_generator = get_db()
    db = next(db_generator)
    with st.sidebar:
        show_sidebar(db)
    
    tab1, tab2, tab3, tab4 = st.tabs(['My investments', 'Tracker', 'Portfolio', 'Calculator'])

    with tab1:
        show_investments_tab(db)
    with tab2:
        show_tracker_tab()
    with tab3:
        # st.info("Portfolio tab – coming soon.")
        show_portfolio_tab()
    with tab4:
        # st.info("Investment calculator – coming soon.")
        show_calculator_tab()


    try:
        db_generator.close()
    except Exception:
        pass

def show_sidebar(db: Session) -> None:
    """
    Renders sidebar with options to add or sell stocks.

    Args:
        db (Session): SQLAlchemy session.
    """
    
    mode = st.radio("Choose action", ["➕ Add Stocks", "📤 Sell Stocks"], horizontal=True)

    if mode == "➕ Add Stocks":
        form_data = investment_form()
        if form_data:
            add_stocks(db, form_data)
            st.success("✅ New investment added successfully!")
            st.experimental_rerun()

    elif mode == "📤 Sell Stocks":
        st.subheader("📤 Sell Your Shares")
        df_wallet = get_wallet(db)
        st.dataframe(df_wallet)

        if not df_wallet.empty:
            selected_company = st.selectbox("Select company", df_wallet['Company'])
            max_shares = int(df_wallet[df_wallet['Company'] == selected_company]['Shares'].max())
            sell_qty = st.number_input("How many shares to sell", min_value=1, max_value=max_shares)

            ticker_symbol = df_wallet[df_wallet['Company'] == selected_company]['Ticker'].values[0]

            try:
                current_price = yf.Ticker(ticker_symbol).history(period="1d")["Close"].iloc[-1]
                st.write(f"💲 Current market price: **{current_price:.2f} USD**")
            except:
                st.warning("⚠️ Could not fetch price data.")
                return

            if st.button(f"Sell {sell_qty} shares of {ticker_symbol}"):
                try:
                    sell_stocks(db, selected_company, sell_qty)
                    st.success(f"✅ Sold {sell_qty} shares of {ticker_symbol} at ${current_price:.2f}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Sale failed: {e}")

def show_investments_tab(db: Session) -> None:
    """
    Renders the 'My investments' tab.
    """

    st.subheader("📤 My Wallet")

    investments = db.query(Investment).filter(Investment.status == "Active").all()

    if not investments:
        st.info("No active investments. Add your first investment!")
        return

    df_investments = pd.DataFrame([{
        "Purchase Date": inv.purchase_date,
        "Company": inv.company_name,
        "Ticker": inv.ticker_symbol,
        "Shares": inv.shares,
        "Purchase Price (USD)": inv.purchase_price_usd,
        "Account Type": inv.account_type,
        "Status": inv.status
    } for inv in investments])

    df_investments["Investment Value"] = df_investments["Shares"] * df_investments["Purchase Price (USD)"]
    df_wallet = df_investments[df_investments["Status"] == "Active"] \
        .groupby(['Ticker', 'Company']).agg({'Shares': 'sum', 'Investment Value': 'sum'}).reset_index()

    st.dataframe(df_wallet)

    with st.expander("📋 Transaction History", expanded=False):
        st.dataframe(df_investments)

    # Summary metrics
    total_value = df_investments["Investment Value"].sum()
    total_shares = df_investments["Shares"].sum()
    profit = calculate_profit(investments)

    col1, col2, col3 = st.columns(3)
    col1.metric("💼 Portfolio Value", f"{total_value:,.2f} USD")
    col2.metric("📈 Estimated Profit", f"{profit:,.2f} USD")
    col3.metric("📊 Total Shares", total_shares)

    st.plotly_chart(plot_investment_value(df_investments), use_container_width=True)


def calculate_profit(investments: list[Investment]) -> float:
    """
    Calculates unrealized profit based on current market price.

    Args:
        investments (list[Investment]): List of active investments

    Returns:
        float: Estimated profit
    """
    profit = 0.0
    for inv in investments:
        try:
            price = yf.Ticker(inv.ticker_symbol).history(period="1d")["Close"].iloc[-1]
            profit += (price - float(inv.purchase_price_usd)) * inv.shares
        except:
            continue
    return profit


def show_tracker_tab() -> None:
    """
    Renders the stock tracker tab with chart and financial metrics.
    """
    # --- Tickers ---
    tickers = get_sp500_tickers()
    ticker = st.selectbox("Select a stock ticker", tickers)

    # --- Time Range ---
    period_choice = ""
    st.markdown("Choose Time Range for the stock price chart")
    periods = ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"]
    cols = st.columns(len(periods))
    for i, p in enumerate(periods):
        if cols[i].button(p):
            period_choice = p

    # --- Candlestick Chart ---
    if ticker and period_choice:
        st.subheader("📉 Candlestick Chart")
        st.plotly_chart(plot_price_chart_candles(ticker, period_choice), use_container_width=True)
    elif ticker:
        st.warning("Please select a time range.")

    # --- Statistics ---
    if ticker:
        st.subheader(f"💡 Key Financial Metrics: {ticker}")
        df_metrics = pd.DataFrame(get_key_metrics(ticker).items(), columns=['Metric', 'Value'])
        st.table(df_metrics)

        st.subheader("📊 Moving Averages")
        st.plotly_chart(plot_moving_averages(ticker), use_container_width=True)

        t = yf.Ticker(ticker)
        st.subheader("📑 Income Statement")
        st.dataframe(format_statement(t.income_stmt))
        st.subheader("📑 Balance Sheet")
        st.dataframe(format_statement(t.balance_sheet))
        st.subheader("📑 Cash Flow")
        st.dataframe(format_statement(t.cash_flow))

        st.plotly_chart(plot_income(format_statement(t.income_stmt)), use_container_width=True)

        st.subheader("📋 Quick Evaluation")
        for line in evaluate_company(get_key_metrics(ticker)):
            st.markdown(line)


import plotly.express as px
import datetime as dt

def show_portfolio_tab() -> None:
    """
    Renders the 'Portfolio Builder' tab to compare multiple stock tickers.
    """
    tickers = get_sp500_tickers()
    selected_tickers = st.multiselect("📊 Select Stocks", options=tickers, placeholder="Search tickers")

    if not selected_tickers:
        st.info("Please select at least one ticker to compare.")
        return

    col1, col2 = st.columns(2)
    start_date = col1.date_input("📅 Start Date", value=dt.date(2023, 1, 1))
    end_date = col2.date_input("📅 End Date", value=dt.date.today())

    if start_date >= end_date:
        st.error("End date must be after start date.")
        return

    # Download data
    try:
        df = yf.download(selected_tickers, start=start_date, end=end_date)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return

    df = df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")
    df.dropna(inplace=True)

    # Calculate % return from start
    df["Start Price"] = df.groupby("Ticker")["Price"].transform("first")
    df["% Change"] = (df["Price"] - df["Start Price"]) / df["Start Price"]

    st.subheader("📈 Normalized Performance Comparison")
    fig = px.line(df, x="Date", y="% Change", color="Ticker", markers=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # Individual stocks summary
    st.subheader("📋 Individual Stock Analysis")
    cols = st.columns(min(3, len(selected_tickers)))

    for i, ticker in enumerate(selected_tickers):
        sub_df = df[df["Ticker"] == ticker]

        try:
            logo_url = f"https://logo.clearbit.com/{yf.Ticker(ticker).info['website'].replace('https://www.', '')}"
            cols[i % 3].image(logo_url, width=65)
        except:
            cols[i % 3].markdown(f"**{ticker}**")

        # Metrics
        recent_prices = sub_df["Price"].tail(365)
        with cols[i % 3]:
            col_metrics = st.columns(3)
            col_metrics[0].metric("📉 50-Day Avg", round(recent_prices.tail(50).mean(), 2))
            col_metrics[1].metric("📉 1-Year Low", round(recent_prices.min(), 2))
            col_metrics[2].metric("📈 1-Year High", round(recent_prices.max(), 2))

            # Plot
            fig = px.line(sub_df, x="Date", y="Price", markers=True)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)


import plotly.graph_objects as go

def show_calculator_tab() -> None:
    """
    Renders the 'Investment Calculator' tab to simulate goal-based portfolio growth.
    """
    tickers = get_sp500_tickers()
    selected_tickers = st.multiselect("📊 Select Stocks", options=tickers, placeholder="Choose tickers for investment")

    if not selected_tickers:
        st.info("Please select at least one ticker.")
        return

    col1, col2 = st.columns(2)
    start_date = col1.date_input("📅 Start Date", dt.date(2023, 1, 1))
    end_date = col2.date_input("📅 End Date", dt.date.today())

    if start_date >= end_date:
        st.error("End date must be after start date.")
        return

    # Investment form
    cols_form = st.columns((1, 3))
    st.markdown("### 💸 Investment Amounts")

    amounts: dict[str, float] = {}
    total_investment = 0

    for ticker in selected_tickers:
        logo_col, input_col = cols_form
        try:
            logo_url = f"https://logo.clearbit.com/{yf.Ticker(ticker).info['website'].replace('https://www.', '')}"
            logo_col.image(logo_url, width=40)
        except:
            logo_col.markdown(f"**{ticker}**")

        amount = input_col.number_input(f"{ticker} Investment Amount", min_value=0, step=50, key=f"amt_{ticker}")
        amounts[ticker] = amount
        total_investment += amount

    # Set goal
    st.markdown("---")
    st.subheader(f"🎯 Total Investment: {total_investment:,.2f} USD")
    goal = st.number_input("🏁 Investment Goal", min_value=0, step=100, value=10000, key="goal")

    # Download prices
    try:
        prices = yf.download(selected_tickers, start=start_date, end=end_date)["Close"]
        prices = prices.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")
    except Exception as e:
        st.error(f"⚠️ Failed to load price data: {e}")
        return

    # Calculate % change per ticker
    prices.dropna(inplace=True)
    prices["Start Price"] = prices.groupby("Ticker")["Price"].transform("first")
    prices["Pct Change"] = (prices["Price"] - prices["Start Price"]) / prices["Start Price"]

    # Apply investment amounts
    prices["Amount"] = prices["Ticker"].map(amounts)
    prices["Investment Value"] = prices["Amount"] * (1 + prices["Pct Change"])

    # Portfolio value
    portfolio = prices.groupby("Date")["Investment Value"].sum().reset_index()
    portfolio["Goal"] = goal

    # Plot
    st.markdown("### 📈 Portfolio Value vs Goal")
    fig = px.area(prices, x="Date", y="Investment Value", color="Ticker")
    fig.add_hline(y=goal, line_color="green", line_dash="dash", line_width=2)

    # When is goal achieved?
    goal_reached = portfolio[portfolio["Investment Value"] >= goal]
    if not goal_reached.empty:
        goal_date = goal_reached.iloc[0]["Date"]
        fig.add_vline(x=goal_date, line_dash="dot", line_color="green")
        fig.add_trace(go.Scatter(
            x=[goal_date + dt.timedelta(days=7)],
            y=[goal * 1.05],
            text=[f"Goal reached: {goal_date.date()}"],
            mode="text",
            textfont=dict(size=16, color="green"),
            showlegend=False
        ))
        st.success(f"✅ Goal will be reached on: {goal_date.date()}")
    else:
        st.warning("❌ The goal is not reached within the selected time range.")

    fig.update_layout(xaxis_title=None, yaxis_title="Value (USD)")
    st.plotly_chart(fig, use_container_width=True)