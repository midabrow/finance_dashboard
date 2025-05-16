# app/components/plots.py

import plotly.express as px
import pandas as pd


import pandas as pd
import plotly.express as px

def plot_expenses_over_time(df: pd.DataFrame) -> px.line:
    """
    Generates a line chart showing total expenses over time (monthly).

    Args:
        df (pd.DataFrame): Cleaned budget dataset.

    Returns:
        plotly.graph_objs._figure.Figure: Line chart of cash flow over time.
    """
    expenses = df[df['Type'] == 'Expense']
    expenses_by_date = expenses.groupby('Date')['Amount'].sum().reset_index()

    fig = px.line(expenses_by_date, x='Date', y='Amount')
    fig.update_layout(xaxis_tickangle=45)
    return fig


def plot_expenses_by_category(df: pd.DataFrame) -> px.pie:
    """
    Generates a pie chart showing the share of expenses by category.

    Args:
        df (pd.DataFrame): Cleaned budget dataset.

    Returns:
        plotly.graph_objs._figure.Figure: Pie chart of expenses by category.
    """
    expenses = df[df['Type'] == 'Expense']
    expenses_by_category = expenses.groupby('Category')['Amount'].sum().reset_index()

    fig = px.pie(expenses_by_category, names='Category', values='Amount')
    fig.update_layout(xaxis_tickangle=45)
    return fig
    

def plot_income_expenses_monthly(df: pd.DataFrame) -> px.bar:
    """
    Generates a grouped bar chart comparing monthly income and expenses.

    Args:
        df (pd.DataFrame): Cleaned budget dataset.

    Returns:
        plotly.graph_objs._figure.Figure: Grouped bar chart of income vs expenses per month.
    """
    income_expenses = df.groupby(["Month", "Type"])['Amount'].sum().reset_index()

    fig = px.bar(income_expenses, x='Month', y='Amount', color='Type', barmode='group')
    return fig


def plot_payment_method_summary(df: pd.DataFrame) -> px.pie:
    """
    Generates a pie chart showing expenses distribution by payment method.

    Args:
        df (pd.DataFrame): Cleaned budget dataset.

    Returns:
        plotly.graph_objs._figure.Figure: Pie chart of expenses by payment method.
    """
    expenses = df[df['Type'] == 'Expense']
    payment_method_summary = expenses.groupby('Payment Method')['Amount'].sum().reset_index()

    fig = px.pie(payment_method_summary, names='Payment Method', values='Amount')
    return fig


def plot_status_summary(df: pd.DataFrame) -> px.pie:
    """
    Generates a pie chart showing the distribution of transaction statuses.

    Args:
        df (pd.DataFrame): Cleaned budget dataset.

    Returns:
        plotly.graph_objs._figure.Figure: Pie chart of transaction statuses.
    """
    status_summary = df.groupby('Status').size().reset_index(name='Count')

    fig = px.pie(status_summary, names='Status', values='Count')
    return fig

def plot_investment_value(df: pd.DataFrame) -> px.bar:
    """
    Generuje wykres wartości inwestycji wg spółek.

    Args:
        df (pd.DataFrame): Dane inwestycji.

    Returns:
        plotly.graph_objs._figure.Figure: Wykres słupkowy wartości inwestycji.
    """
    investment_summary = df.copy()
    investment_summary['Total value (USD)'] = investment_summary['Shares'] * investment_summary['Purchase Price (USD)']

    fig = px.bar(
        investment_summary,
        x='Ticker',
        y='Total value (USD)',
        text_auto='.2s',
        title="Investment Value by Company",
        labels={"Ticker": "Ticker", "Total value (USD)": "Value USD"}
    )

    fig.update_layout(xaxis_tickangle=45)
    return fig