# app/models/budget.py

from sqlalchemy import Column, Integer, String, Numeric, Date, Text, TIMESTAMP
from app.components.database import Base
from datetime import datetime, timezone

class Expense(Base):
    """
    ORM model representing a budget entry (income or expense).
    """

    __tablename__ = "expenses"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    type = Column(String(20), nullable=False) 
    amount = Column(Numeric(12, 2), nullable=False)
    payment_method = Column(String(50))
    status = Column(String(20), default="Completed")
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Expense(id={self.id}, description={self.description}, amount={self.amount})>"
