# app/models/budget.py

from sqlalchemy import Column, Integer, String, Numeric, Date, Text, TIMESTAMP
from components.database import Base
from datetime import datetime

class Expense(Base):
    """
    Model ORM reprezentujący wpis budżetowy (przychód lub wydatek).
    """

    __tablename__ = "expenses"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    type = Column(String(20), nullable=False) 
    amount = Column(Numeric(12, 2), nullable=False)
    payment_method = Column(String(50))
    status = Column(String(20), default="Completed")
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __repr__(self):
        return f"<Expense(id={self.id}, description={self.description}, amount={self.amount})>"
