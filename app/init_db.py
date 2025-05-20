# init_db.py
# Do lokalnej pracy
# python app/init_db.py

from components.database import Base, engine
from models.investment import Investment
from models.budget import Expense

# Utworzenie wszystkich tabel
Base.metadata.create_all(bind=engine)
print("✅ Tabele zostały utworzone w lokalnej bazie danych.")
