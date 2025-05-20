import streamlit as st
import pandas as pd
from components.data_cleaning import BudgetDataCleaner
from components.plots import plot_expenses_over_time, plot_expenses_by_category, plot_income_expenses_monthly, plot_payment_method_summary, plot_status_summary
import os
from utils.styles import STYLES, styled_text, styled_text_with_const_style
from components.forms import expense_form
from components.database import get_db
from models.budget import Expense

def show_home_page():
    """
    Main Page - Home Budget
    """

    st.title("💸 HOME BUDGET")
    st.markdown("---")
    data_source = st.radio("Select data source:", ["Upload CSV", "Load from Database"])

    with st.sidebar:
        st.header("➕ Add Expense or Income")
        form_data = expense_form()

        if form_data:
            db = next(get_db())
            new_expense = Expense(
                date=form_data["date"],
                description=form_data["description"],
                category=form_data["category"],
                type=form_data["type"],
                amount=form_data["amount"],
                payment_method=form_data["payment_method"],
                status=form_data["status"],
                created_at=form_data["created_at"]
            )
            db.add(new_expense)
            db.commit()
            st.success("✅ Entry successfully added to database.")

    if data_source == "Upload CSV":

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

            # Zapisz do bazy
            save_button = st.button("Save cleaned data to database")
            if save_button:
                db = next(get_db())
                for _, row in df_clean.iterrows():
                    expense = Expense(
                        date=row["Date"],
                        description=row["Description"],
                        category=row["Category"],
                        type=row["Type"],
                        amount=row["Amount"],
                        payment_method=row["Payment Method"],
                        status=row["Status"]
                    )
                    db.add(expense)
                db.commit()
                st.success("✅ Data imported and saved to database.")

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
            status = df_clean['Status'].unique().tolist()


            col1, col2, col3 = st.columns(3)
            with col1:
                selected_year = st.selectbox("Year", options=sorted(years))
                selected_month = st.selectbox("Month", options=sorted(months))
            with col2:
                selected_category = st.selectbox("Category", options=["All"] + sorted(categories))
                selected_type = st.selectbox("Type", options=["All"] + sorted(types))
            with col3:
                selected_payment_method = st.selectbox("Payment Method", options=["All"] + sorted(payment_method))
                selected_status = st.selectbox("Status", options=["All"] + sorted(status))

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

            if selected_status != "All":
                df_filtered = df_filtered[df_filtered['Status'] == selected_status]


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

    elif data_source == "Load from Database":
        db = next(get_db())
        records = db.query(Expense).all()

        if not records:
            st.warning("⚠️ No records found in the database.")
            return

        # Konwersja do DataFrame
        df_clean = pd.DataFrame([{
            "Date": r.date,
            "Description": r.description,
            "Category": r.category,
            "Type": r.type,
            "Amount": float(r.amount),
            "Payment Method": r.payment_method,
            "Status": r.status,
            "Created At": r.created_at
        } for r in records])
        
        df_clean["Year"] = pd.to_datetime(df_clean["Date"]).dt.year
        df_clean["Month"] = pd.to_datetime(df_clean["Date"]).dt.month

        
        st.success("✅ Data loaded from database.")
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
        status = df_clean['Status'].unique().tolist()


        col1, col2, col3 = st.columns(3)
        with col1:
            selected_year = st.selectbox("Year", options=sorted(years))
            selected_month = st.selectbox("Month", options=sorted(months))
        with col2:
            selected_category = st.selectbox("Category", options=["All"] + sorted(categories))
            selected_type = st.selectbox("Type", options=["All"] + sorted(types))
        with col3:
            selected_payment_method = st.selectbox("Payment Method", options=["All"] + sorted(payment_method))
            selected_status = st.selectbox("Status", options=["All"] + sorted(status))

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

        if selected_status != "All":
            df_filtered = df_filtered[df_filtered['Status'] == selected_status]


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
