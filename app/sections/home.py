import streamlit as st
import pandas as pd
import os
from components.forms import expense_form
from components.database import get_db
from components.data_cleaning import BudgetDataCleaner
from components.plots import (
    plot_expenses_over_time,
    plot_expenses_by_category,
    plot_income_expenses_monthly,
    plot_payment_method_summary,
    plot_status_summary,
)
from models.budget import Expense
from services.budget_actions import save_expense, save_expense_dataframe
from utils.styles import STYLES, styled_text, styled_text_with_const_style


def show_home_page() -> None:
    """
    Main function for the Home Budget page.
    Allows user to add entries, upload CSVs or load from database, and visualize budget data.
    """
    st.title("💸 HOME BUDGET")
    st.markdown("---")

    data_source = st.radio("Select data source:", ["Upload CSV", "Load from Database"])

    with st.sidebar:
        st.header("➕ Add Expense or Income")
        form_data = expense_form()
        if form_data:
            db = next(get_db())
            save_expense(db, form_data)
            st.success("✅ Entry successfully added to database.")

    if data_source == "Upload CSV":
        df = upload_and_clean_csv()
        if df is not None:
            render_dashboard(df)

    elif data_source == "Load from Database":
        df = load_data_from_db()
        if df is not None:
            render_dashboard(df)

    else:
        st.info("Upload a file to start analyzing your budget.")


def upload_and_clean_csv() -> pd.DataFrame | None:
    """
    Handles CSV upload and data cleaning.

    Returns:
        pd.DataFrame | None: Cleaned DataFrame or None if no file is uploaded
    """
    st.subheader("📤 Import Your Budget Data")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if not uploaded_file:
        return None

    temp_path = "temp_data.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cleaner = BudgetDataCleaner(temp_path)
    df = cleaner.clean_data()

    if st.button("💾 Save cleaned data to database"):
        db = next(get_db())
        save_expense_dataframe(db, df)
        st.success("✅ Data saved to database.")

    return df


def load_data_from_db() -> pd.DataFrame | None:
    """
    Loads budget records from the database.

    Returns:
        pd.DataFrame | None: DataFrame of records or None if empty
    """
    db = next(get_db())
    records = db.query(Expense).all()
    if not records:
        st.warning("⚠️ No records found in the database.")
        return None

    df = pd.DataFrame([{
        "Date": r.date,
        "Description": r.description,
        "Category": r.category,
        "Type": r.type,
        "Amount": float(r.amount),
        "Payment Method": r.payment_method,
        "Status": r.status,
        "Created At": r.created_at
    } for r in records])

    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["Month"] = pd.to_datetime(df["Date"]).dt.month

    st.success("✅ Data loaded from database.")
    return df


def render_dashboard(df: pd.DataFrame) -> None:
    """
    Renders filters, statistics, and charts for budget data.

    Args:
        df (pd.DataFrame): Budget DataFrame
    """
    st.subheader("📊 General Statistics")
    total_income = df[df["Type"] == "Income"]["Amount"].sum()
    total_expense = df[df["Type"] == "Expense"]["Amount"].sum()
    balance = total_income - total_expense

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"{total_income:,.2f} PLN", delta_color="normal", border=True)
    col2.metric("Total Expenses", f"{total_expense:,.2f} PLN", delta_color="inverse", border=True)
    col3.metric("Net Balance", f"{balance:,.2f} PLN", border=True)

    df_filtered = filter_dataframe(df)

    st.subheader("📋 Transaction Overview")
    st.dataframe(df_filtered)

    st.markdown(styled_text_with_const_style("📊 Budget Overview", STYLES["title"]), unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown(styled_text("📈 Expenses Over Time", "#10B981"), unsafe_allow_html=True)
        st.plotly_chart(plot_expenses_over_time(df_filtered), use_container_width=True)

    with row1_col2:
        st.markdown(styled_text("📊 Expenses by Category", "#F59E0B"), unsafe_allow_html=True)
        st.plotly_chart(plot_expenses_by_category(df_filtered), use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown(styled_text("📦 Transaction Status", "#8B5CF6"), unsafe_allow_html=True)
        st.plotly_chart(plot_status_summary(df_filtered), use_container_width=True)
    with row2_col2:
        st.markdown(styled_text("💳 Payment Method Breakdown", "#EC4899"), unsafe_allow_html=True)
        st.plotly_chart(plot_payment_method_summary(df_filtered), use_container_width=True)

    st.markdown(styled_text("💰 Monthly Income vs Expenses", "#3B82F6"), unsafe_allow_html=True)
    st.plotly_chart(plot_income_expenses_monthly(df_filtered), use_container_width=True)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the input DataFrame based on user input from sidebar widgets.

    Args:
        df (pd.DataFrame): Full DataFrame

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    st.subheader("🔍 Filter Data")
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["Month"] = pd.to_datetime(df["Date"]).dt.month

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_year = st.selectbox("Year", sorted(df["Year"].unique()))
        selected_month = st.selectbox("Month", sorted(df["Month"].unique()))
    with col2:
        selected_category = st.selectbox("Category", ["All"] + sorted(df["Category"].unique()))
        selected_type = st.selectbox("Type", ["All"] + sorted(df["Type"].unique()))
    with col3:
        selected_payment = st.selectbox("Payment Method", ["All"] + sorted(df["Payment Method"].unique()))
        selected_status = st.selectbox("Status", ["All"] + sorted(df["Status"].unique()))

    df_filtered = df[
        (df["Year"] == selected_year) &
        (df["Month"] == selected_month)
    ]

    if selected_category != "All":
        df_filtered = df_filtered[df_filtered["Category"] == selected_category]
    if selected_type != "All":
        df_filtered = df_filtered[df_filtered["Type"] == selected_type]
    if selected_payment != "All":
        df_filtered = df_filtered[df_filtered["Payment Method"] == selected_payment]
    if selected_status != "All":
        df_filtered = df_filtered[df_filtered["Status"] == selected_status]

    return df_filtered
