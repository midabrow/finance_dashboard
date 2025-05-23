import pytest
from datetime import datetime, timezone
from app.services.budget_actions import save_expense, save_expense_dataframe
from app.models.budget import Expense
import pandas as pd

def test_save_expense_creates_entry(db_session):
    form_data = {
        "date": "2024-01-10",
        "description": "Fuel",
        "category": "Transport",
        "type": "Expense",
        "amount": 150.00,
        "payment_method": "Card",
        "status": "Completed",
        "created_at": datetime(2024, 1, 10, tzinfo=timezone.utc)
    }

    save_expense(db_session, form_data)
    result = db_session.query(Expense).filter_by(description="Fuel").first()

    assert result is not None
    assert result.amount == 150.00
    assert result.category == "Transport"
    assert result.status == "Completed"

def test_save_expense_dataframe_creates_multiple_entries(db_session):
    data = {
        "Date": ["2024-01-01", "2024-01-02"],
        "Description": ["Book", "Groceries"],
        "Category": ["Education", "Food"],
        "Type": ["Expense", "Expense"],
        "Amount": [40.0, 200.0],
        "Payment Method": ["Cash", "Card"],
        "Status": ["Completed", "Completed"]
    }
    df = pd.DataFrame(data)
    
    save_expense_dataframe(db_session, df)
    results = db_session.query(Expense).all()

    assert len(results) == 2
    assert results[0].description == "Book"
    assert results[1].amount == 200.0
