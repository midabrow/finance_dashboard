from sqlalchemy.orm import Session
from models.budget import Expense
import pandas as pd
from typing import Dict


def save_expense(db: Session, form_data: Dict) -> None:
    """
    Adds a single expense or income record to the database.

    Args:
        db (Session): SQLAlchemy session
        form_data (dict): Data from expense_form()
    """
    expense = Expense(
        date=form_data["date"],
        description=form_data["description"],
        category=form_data["category"],
        type=form_data["type"],
        amount=form_data["amount"],
        payment_method=form_data["payment_method"],
        status=form_data["status"],
        created_at=form_data["created_at"]
    )
    db.add(expense)
    db.commit()


def save_expense_dataframe(db: Session, df: pd.DataFrame) -> None:
    """
    Saves all rows from a cleaned DataFrame to the database.

    Args:
        db (Session): SQLAlchemy session
        df (pd.DataFrame): Cleaned expense data
    """
    for _, row in df.iterrows():
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
