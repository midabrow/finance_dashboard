# app/utils/stock_api.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from typing import Optional
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st

def plot_price_chart_candles(ticker:str, period: str ="6mo") -> go.Figure:
    df = yf.download(ticker, period=period)

    if df.empty:
        st.error("❌ Brak danych do wykresu świecowego.")
        return go.Figure()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)


    fig = go.Figure(
        data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick')]
    )
    fig.update_layout(
        title=f"{ticker} - Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False
    )
    return fig

def plot_moving_averages(ticker: str, period: str = "6mo") -> go.Figure:
    """
    Tworzy wykres średnich kroczących SMA50 i EMA20 jako osobny wykres liniowy.
    """
    df = yf.download(ticker, period=period)
    if df.empty:
        st.error("❌ Brak danych do wykresu średnich kroczących.")
        return go.Figure()

    df.index = pd.to_datetime(df.index)
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    ma_df = df[["SMA50", "EMA20"]].dropna().reset_index()
    ma_df = ma_df.rename(columns={"index": "Date"})

    fig = px.line(ma_df, x="Date", y=["SMA50", "EMA20"],
                  title=f"{ticker} - SMA50 vs EMA20",
                  labels={"value": "Price ($)", "variable": "Moving Average"})
    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)")

    return fig

def get_key_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info

    return {
        "Current Price": info.get("currentPrice"),
        "Previous Close": info.get("previousClose"),
        "Market Cap": info.get("marketCap"),
        "P/E (TTM)": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "EPS (TTM)": info.get("trailingEps"),
        "Dividend Yield": info.get("dividendYield"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Debt to Equity": info.get("debtToEquity"),
        "Price to Book": info.get("priceToBook")
    }

def format_statement(df):
    df = df.T
    df.index = df.index.strftime('%Y-%m-%d')
    return df.iloc[::-1]


def plot_income(data):
    try:
        # Resetuj indeks i zamień datę na kolumnę
        df = data[["Total Revenue", "Gross Profit", "Net Income"]].copy()
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})

        # Melt do formatu long
        df_melted = df.melt(id_vars="Date", 
                            value_vars=["Total Revenue", "Gross Profit", "Net Income"], 
                            var_name="Metric", 
                            value_name="Value")

        # Wykres słupkowy
        fig = px.bar(
            df_melted,
            x="Date",
            y="Value",
            color="Metric",
            barmode="group",
            title="📊 Revenue, Profit & Net Income",
            labels={"Value": "USD", "Metric": "Metric"}
        )
        fig.update_layout(xaxis=dict(type="category"))

        return fig
    except KeyError as e:
        st.error(f"Błąd danych: {e}")
        return go.Figure()
    
def evaluate_company(metrics):
    messages = []
    if metrics["P/E (TTM)"] and metrics["P/E (TTM)"] < 15:
        messages.append("✅ Attractive valuation (P/E < 15)")
    else:
        messages.append("⚠️ P/E may be high")
        
    if metrics["ROE"] and metrics["ROE"] > 0.15:
        messages.append("✅ High Return on Equity (ROE > 15%)")

    if metrics["Debt to Equity"] and metrics["Debt to Equity"] < 1:
        messages.append("✅ Reasonable debt level (D/E < 1)")

    return messages