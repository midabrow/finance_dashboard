# app/pages/investments.py

import streamlit as st
from sqlalchemy.orm import Session
from components.database import get_db
from models.investment import Investment
from components.forms import investment_form
from components.plots import plot_investment_value
import pandas as pd
from utils.styles import styled_text

def show_investments_page():
    """
    Page – Investment Portfolio.
    """

    st.title("📈 Investment Portfolio")
    st.markdown("---")

    # Create database session
    db_generator = get_db()
    db: Session = next(db_generator)

    # Add new investment
    st.markdown(styled_text("➕ Add New Investment", color="#10B981", font_size="18px", margin_bottom="1rem"), unsafe_allow_html=True)
    new_investment = investment_form()

    if new_investment:
        investment = Investment(**new_investment)
        db.add(investment)
        db.commit()
        st.success("✅ New investment added successfully!")

    st.markdown("---")



    # Close DB session
    try:
        db_generator.close()
    except:
        pass
