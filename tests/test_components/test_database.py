from app.models.budget import Expense
from app.models.investment import Investment

def test_insert_expense(db_session):
    # Arrange
    new_expense = Expense(
        date="2024-01-01",
        description="Laptop",
        category="Electronics",
        type="Expense",
        amount=4500.00,
        payment_method="Credit Card",
        status="Completed"
    )

    # Act – zapis do bazy
    db_session.add(new_expense)
    db_session.commit()

    # Assert – sprawdzenie, czy się zapisało
    result = db_session.query(Expense).filter_by(description="Laptop").first()
    assert result is not None
    assert result.amount == 4500.00
    assert result.category == "Electronics"


def test_get_expense(db_session):
    entry = Expense(
        date="2024-02-10",
        description="Book",
        category="Education",
        type="Expense",
        amount=59.99,
        payment_method="Cash"
    )
    db_session.add(entry)
    db_session.commit()

    expenses = db_session.query(Expense).all()
    assert len(expenses) == 1
    assert expenses[0].description == "Book"


def test_insert_investment(db_session):
    investment = Investment(
        purchase_date="2024-01-15",
        company_name="NVIDIA Corporation",
        ticker_symbol="NVDA",
        shares=10,
        purchase_price_usd=490.75,
        account_type="IKE"
    )

    db_session.add(investment)
    db_session.commit()

    result = db_session.query(Investment).filter_by(ticker_symbol="NVDA").first()
    assert result is not None
    assert result.company_name == "NVIDIA Corporation"
    assert result.shares == 10
    assert result.status == "Active"

def test_update_investment_sell(db_session):
    inv = Investment(
        purchase_date="2024-01-01",
        company_name="Apple Inc.",
        ticker_symbol="AAPL",
        shares=5,
        purchase_price_usd=175.50,
        account_type="IKZE"
    )
    db_session.add(inv)
    db_session.commit()

    # Sprzedaj akcje
    inv.sell_price_usd = 190.25
    inv.sold_date = "2024-04-01"
    inv.status = "Sold"
    db_session.commit()

    updated = db_session.query(Investment).filter_by(ticker_symbol="AAPL").first()
    assert updated.sell_price_usd == 190.25
    assert updated.status == "Sold"