# app/components/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_DB = os.getenv("POSTGRES_DB", "finance_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "finance_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "supersecure")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# postgresql://my_user:super_secret@localhost:5432/my_database
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create database engine
engine = create_engine(DATABASE_URL, echo=False)

# Session to communicate with DB
localSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

def get_db():
    db = localSession()
    try:
        yield db
    finally:
        db.close()