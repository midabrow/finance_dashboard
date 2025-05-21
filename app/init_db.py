# init_db.py
# Do lokalnej pracy
# python app/init_db.py

from components.database import Base, engine
from models.investment import Investment
from models.budget import Expense

Base.metadata.drop_all(bind=engine)   # ❗ usunie wszystkie tabele
Base.metadata.create_all(bind=engine) # utworzy je na nowo z nowymi polami
print("✅ Tabele zostały utworzone w lokalnej bazie danych.")
