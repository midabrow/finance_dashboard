# app/models/investment.py

from sqlalchemy import Column, Integer, String, Numeric, Date, Text, TIMESTAMP
from app.components.database import Base
from datetime import datetime, timezone

class Investment(Base):
    """
    ORM model representing a single stock investment.
    """

    __tablename__ = "investments"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    purchase_date = Column(Date, nullable=False)
    company_name = Column(Text, nullable=False)
    ticker_symbol = Column(String(10), nullable=False, index=True)
    shares = Column(Integer, nullable=False)
    purchase_price_usd = Column(Numeric(12, 2), nullable=False)
    sold_date = Column(Date, nullable=True)
    sell_price_usd = Column(Numeric(12, 2), nullable=True)
    account_type = Column(String(20), default="Standard") 
    currency = Column(String(5), default="USD")
    status = Column(String(20), default="Active")
    created_at = Column(TIMESTAMP, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Investment(id={self.id}, ticker={self.ticker_symbol}, shares={self.shares})>"
