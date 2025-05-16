import streamlit as st
import pandas as pd
from components.data_cleaning import BudgetDataCleaner
from components.plots import plot_expenses_over_time, plot_expenses_by_category, plot_income_expenses_monthly, plot_payment_method_summary, plot_status_summary
import os
from utils.styles import STYLES, styled_text, styled_text_with_const_style

def show_home_page():
    """
    Main Page - Home Budget
    """

    st.title("💸 HOME BUDGET")
    st.markdown("---")

    st.subheader("Import Your Budget Data")
    uploaded_file = st.file_uploader("Upload a CSV file with your budget", type="csv")


    if uploaded_file:
        # Zapisanie pliku tymczasowo
        temp_file_path = os.path.join("temp_data.csv")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Cleaning data
        cleaner = BudgetDataCleaner(filepath=temp_file_path)
        df_clean = cleaner.clean_data()
        st.dataframe(df_clean)

        # Szybkie statystyki
        st.subheader("📊 General Statistics")
        total_income = df_clean[df_clean['Type'] == "Income"]['Amount'].sum()
        total_expense = df_clean[df_clean['Type'] == "Expense"]['Amount'].sum()
        balance = total_income - total_expense


        col1, col2, col3 = st.columns(3)
        col1.metric(label=":green[**Total Income**]", value=f"{total_income:,.2f} PLN", border=True)
        col2.metric(label=":red[**Total Expenses**]", value=f"{total_expense:,.2f} PLN", border=True)
        col3.metric(label=":gray[**Net Balance**]", value=f"{balance:,.2f} PLN", delta_color="inverse" if balance < 0 else "normal", border=True)

        # Filters
        st.subheader("🔍 Filter Data")
        years = df_clean['Year'].unique().tolist()
        months = df_clean['Month'].unique().tolist()
        categories = df_clean['Category'].unique().tolist()
        types = df_clean['Type'].unique().tolist()
        payment_method = df_clean['Payment Method'].unique().tolist()

        selected_year = st.selectbox("Year", options=sorted(years))
        selected_month = st.selectbox("Month", options=sorted(months))
        selected_category = st.selectbox("Category", options=["All"] + sorted(categories))
        selected_type = st.selectbox("Type", options=["All"] + sorted(types))
        selected_payment_method = st.selectbox("Payment Method", options=["All"] + sorted(payment_method))

        df_filtered = df_clean[
            (df_clean['Year'] == selected_year) &
            (df_clean['Month'] == selected_month)
        ]

        if selected_category != "All":
            df_filtered = df_filtered[df_filtered['Category'] == selected_category]

        if selected_type != "All":
            df_filtered = df_filtered[df_filtered['Type'] == selected_type]

        if selected_payment_method != "All":
            df_filtered = df_filtered[df_filtered['Payment Method'] == selected_payment_method]


        st.subheader("📋 Transaction Overview")
        st.dataframe(df_filtered)

        # Charts
        st.markdown(styled_text_with_const_style("📊 Budget Overview", STYLES["title"]), unsafe_allow_html=True)

        # Row 1
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown(styled_text("📈 Expenses Over Time", color="#10B981"), unsafe_allow_html=True)
            st.plotly_chart(plot_expenses_over_time(df_clean), use_container_width=True)
        with row1_col2:
            st.markdown(styled_text("📊 Expenses by Category", color="#F59E0B"), unsafe_allow_html=True)
            st.plotly_chart(plot_expenses_by_category(df_clean), use_container_width=True)

        # Row 2
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.markdown(styled_text("📦 Transaction Status", color="#8B5CF6"), unsafe_allow_html=True)
            st.plotly_chart(plot_status_summary(df_clean), use_container_width=True)
        with row2_col2:
            st.markdown(styled_text("💳 Payment Method Breakdown", color="#EC4899"), unsafe_allow_html=True)
            st.plotly_chart(plot_payment_method_summary(df_clean), use_container_width=True)
        
        st.markdown(styled_text("💰 Monthly Income vs Expenses", color="#3B82F6"), unsafe_allow_html=True)
        st.plotly_chart(plot_income_expenses_monthly(df_clean), use_container_width=True)


    else:
        st.info("Upload your CSV file to start analyzing your budget!")
