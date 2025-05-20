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

        # st.subheader("🔎 Search for a Company")

        # sidebar_text = '''
        # <div style="background-color: #82c7fd; padding: 20px; border-radius: 10px; text-align: center;">
        #     <h1 style="color: #000; font-size: 24px;">Stock Price Data and Statistics</h1>
        # </div>

        # ## About This App

        # This application allows users to visualise stock price data, statistics and financial data. 

        # ### Disclaimer

        # - The information provided is for educational and demonstration purposes only.
        # - This is not financial advice.

        # '''
        # st.sidebar.markdown(sidebar_text, unsafe_allow_html=True)

def show_stock_tracker_page():
    """
    Page – Stock Price Tracker.
    """

    st.title("📊 Stock Price Tracker")

    
    with st.sidebar:
        st.header("➕ Add Stocks")
        form_data = investment_form()

        if form_data:
            db = next(get_db())
            new_investment = Investment(
                    purchase_date = form_data["purchase_date"],
                    company_name = form_data["company_name"],
                    ticker_symbol = form_data["ticker_symbol"],
                    shares = form_data["shares"],
                    purchase_price_usd = form_data["purchase_price_usd"],
                    account_type = form_data["account_type"],
                    currency = form_data["currency"],
                    status = form_data["status"]
            )
            if new_investment:
                db.add(new_investment)
                db.commit()
                st.success("✅ New investment added successfully!")

        try:
            db_generator.close()
        except:
            pass
    tab1, tab2, tab3, tab4 = st.tabs(['My investments', 'Tracker', 'Portfolio', 'Calculator'])

    with tab1:
        # Display existing investments
        # st.markdown(styled_text("📋 Your Investments", color="#3B82F6", font_size="18px", margin_bottom="1rem"), unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Create database session
        db_generator = get_db()
        db: Session = next(db_generator)

        investments = db.query(Investment).filter(Investment.status == "Active").all()

        investment_amount = sum(inv.shares * float(inv.purchase_price_usd) for inv in investments)

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

            # Close DB session
            try:
                db_generator.close()
            except:
                pass
            
            profit = 0.0

            for inv in investments:
                try:
                    ticker = yf.Ticker(inv.ticker_symbol)
                    current_price = ticker.history(period="1d")["Close"].iloc[-1]

                    total_purchase = inv.shares * float(inv.purchase_price_usd)
                    total_current_value = inv.shares * current_price
                    profit += total_current_value - total_purchase
                except Exception as e:
                    st.warning(f"⚠️ Could not fetch data for {inv.ticker_symbol}: {e}")

            # Basic stats
            total_shares = sum(inv.shares for inv in investments)
            total_investment_value = sum(inv.shares * float(inv.purchase_price_usd) for inv in investments)
            
            with col1:
                col1.metric(label=":green[**💵 Portfolio Value (USD)**]", value=f"{investment_amount:,.2f} USD", border=True)
                col1.metric("📈 Total Shares", total_shares, border=True)
            with col2:
                col2.metric(label=":green[**Profit**]", value=f"{profit:,.2f} USD", border=True)

            # Plot by company
            st.markdown(styled_text("📊 Investment Value by Company", color="#F59E0B", font_size="16px", margin_top="1.5rem"), unsafe_allow_html=True)
            st.plotly_chart(plot_investment_value(df_investments), use_container_width=True)

        else:
            st.info("No active investments found. Add your first investment! 🚀")

    with tab2:

        
        # --- Tickers ---
        tickers = get_sp500_tickers()
        ticker = st.selectbox("Select a stock ticker", tickers)

        # --- Time Range ---
        period_options = ["1d", "5d", "1mo", "6mo", "1y", "5y", "max"]
        period_choice = ""

        st.markdown("Choose Time Range for the stock price chart")
        one_day, five_days, one_month, six_month, one_year, five_year = st.columns(6)

        if one_day.button(period_options[0], use_container_width=True):
            period_choice = period_options[0]
        if five_days.button(period_options[1], use_container_width=True):
            period_choice = period_options[1]
        if one_month.button(period_options[2], use_container_width=True):
            period_choice = period_options[2]
        if six_month.button(period_options[3], use_container_width=True):
            period_choice = period_options[3]
        if one_year.button(period_options[4], use_container_width=True):
            period_choice = period_options[4]
        if five_year.button(period_options[5], use_container_width=True):
            period_choice = period_options[5]

        
        # --- Candlestick Chart ---
        if ticker and period_choice:
            st.write(period_choice)
            st.subheader("📉 Candlestick Chart")
            candle_fig = plot_price_chart_candles(ticker, period_choice)
            st.plotly_chart(candle_fig, use_container_width=True)

            st.success("✅ Chart successfully loaded!")

        elif ticker and not period_choice:
            st.warning("Please select a time range.")
        else:
            st.info("Enter a stock ticker to display the chart.")


        # --- Statistics ---
        if ticker:
            st.subheader(f"💡 Key Financial Metrics: {ticker}")
            metrics = get_key_metrics(ticker)
            df_metrics = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
            st.table(df_metrics)

            st.subheader("📊 Moving Averages: SMA50 and EMA20")
            ma_fig = plot_moving_averages(ticker)
            st.plotly_chart(ma_fig, use_container_width=True)

            t = yf.Ticker(ticker)
            income = format_statement(t.income_stmt)
            balance = format_statement(t.balance_sheet)
            cashflow = format_statement(t.cash_flow)

            st.subheader("📑 Income Statement")
            st.dataframe(income)
            st.subheader("📑 Balance Sheet")
            st.dataframe(balance)
            st.subheader("📑 Cash Flow")
            st.dataframe(cashflow)


            st.plotly_chart(plot_income(income), use_container_width=True)

            st.subheader("📋 Quick Evaluation Summary")
            for line in evaluate_company(metrics):
                st.markdown(line)

    with tab3:
        selected_ticker = st.multiselect('Portfolio Builder', placeholder="Search tickers", options=tickers)

        # cols = st.columns(4)
        # for i, ticker in enumerate(selected_ticker):
        #     try:
        #         cols[i % 4].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
        #     except:
        #         cols[i % 4].subheader(ticker)

        # # Date selector
        # cols = st.columns(2)
        # sel_dt1 = cols[0].date_input('Start Date', value=dt.datetime(2024,1,1), format='YYYY-MM-DD')
        # sel_dt2 = cols[1].date_input('End Date', format='YYYY-MM-DD')

        # st.subheader('All Stocks')
        #     # Select tickers data
        
        # if len(selected_ticker) != 0:
        #     yfdata = yf.download(list(selected_ticker), start=sel_dt1, end=sel_dt2)['Close'].reset_index().melt(id_vars = ['Date'], var_name = 'ticker', value_name='price')
        #     yfdata['price_start'] = yfdata.groupby('ticker').price.transform('first')
        #     yfdata['price_pct_daily'] = yfdata.groupby('ticker').price.pct_change()
        #     yfdata['price_pct'] = (yfdata.price - yfdata.price_start) / yfdata.price_start
        # fig = px.line(yfdata, x='Date', y='price_pct', color='ticker', markers=True)
        # fig.add_hline(y=0, line_dash="dash", line_color="white") 
        # fig.update_layout(xaxis_title=None, yaxis_title=None)
        # fig.update_yaxes(tickformat=',.0%') 
        # st.plotly_chart(fig, use_container_width=True)

        # # Individual stock plots
        # st.subheader('Individual Stock')
        # cols = st.columns(3)
        # for i, ticker in enumerate(selected_ticker):
        #     # Adding logo
        #     try:
        #         cols[i % 3].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
        #     except:
        #         cols[i % 3].subheader(ticker)

        #     # Stock metrics
        #     cols2 = cols[i % 3].columns(3)
        #     ticker = 'Close' if len(selected_ticker) == 1 else ticker
        #     cols2[0].metric(label='50-Day Average', value=round(yfdata[yfdata.ticker == ticker].price.tail(50).mean(),2))
        #     cols2[1].metric(label='1-Year Low', value=round(yfdata[yfdata.ticker == ticker].price.tail(365).min(),2))
        #     cols2[2].metric(label='1-Year High', value=round(yfdata[yfdata.ticker == ticker].price.tail(365).max(),2))

        #     # Stock plot
        #     fig = px.line(yfdata[yfdata.ticker == ticker], x='Date', y='price', markers=True)
        #     fig.update_layout(xaxis_title=None, yaxis_title=None)
        #     cols[i % 3].plotly_chart(fig, use_container_width=True)

    with tab4:
        cols_tab2 = st.columns((0.2,0.8))
        total_inv = 0
        amounts = {}
        for i, ticker in enumerate(selected_ticker):
            cols = cols_tab2[0].columns((0.1,0.3))
            try:
                cols[0].image('https://logo.clearbit.com/' + yf.Ticker(ticker).info['website'].replace('https://www.', ''), width=65)
            except:
                cols[0].subheader(ticker)

            amount = cols[1].number_input('', key=ticker, step=50)
            total_inv = total_inv + amount
            amounts[ticker] = amount

        # # Investment goals
        # cols_tab2[1].subheader('Total Investment: ' + str(total_inv))
        # cols_goal = cols_tab2[1].columns((0.06,0.20,0.7))
        # cols_goal[0].text('')
        # cols_goal[0].subheader('Goal: ')
        # goal = cols_goal[1].number_input('', key='goal', step=50)

        # # Plot
        # df = yfdata.copy()
        # df['amount'] = df.ticker.map(amounts) * (1 + df.price_pct)

        # dfsum = df.groupby('Date').amount.sum().reset_index()
        # fig = px.area(df, x='Date', y='amount', color='ticker')
        # fig.add_hline(y=goal, line_color='rgb(57,255,20)', line_dash='dash', line_width=3)
        # if dfsum[dfsum.amount >= goal].shape[0] == 0:
        #     cols_tab2[1].warning("The goal can't be reached within this time frame. Either change the goal amount or the time frame.")
        # else:
        #     fig.add_vline(x=dfsum[dfsum.amount >= goal].Date.iloc[0], line_color='rgb(57,255,20)', line_dash='dash', line_width=3)
        #     fig.add_trace(go.Scatter(x=[dfsum[dfsum.amount >= goal].Date.iloc[0] + dt.timedelta(days=7)], y=[goal*1.1], text=[dfsum[dfsum.amount >= goal].Date.dt.date.iloc[0]], mode='text', name="Goal", textfont=dict(color='rgb(57,255,20)', size=20)))
        # fig.update_layout(xaxis_title=None, yaxis_title=None)
        # cols_tab2[1].plotly_chart(fig, use_container_width=True)   