# tests/conftest.py

import pytest
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.components.database import Base
from app.models.budget import Expense
from app.models.investment import Investment

from dotenv import load_dotenv

@pytest.fixture
def sample_income_df():
    df = pd.DataFrame({
        "2023-12-31": [1000, 800, 200],
        "2022-12-31": [950, 760, 190]
    }, index=["Total Revenue", "Gross Profit", "Net Income"])

    df.columns = pd.to_datetime(df.columns)
    return df


load_dotenv(dotenv_path=".env.test")
print("✅ LOADED ENV VALUES:")
print("  DB =", os.getenv("DB_NAME"))
print("  URL =", f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")

POSTGRES_DB = os.getenv("DB_NAME")
POSTGRES_USER = os.getenv("DB_USER")
POSTGRES_PASSWORD = os.getenv("DB_PASSWORD")
POSTGRES_HOST = os.getenv("DB_HOST", "localhost")
POSTGRES_PORT = os.getenv("DB_PORT", "5432")

TEST_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(TEST_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def db_engine():
    return engine

@pytest.fixture(scope="function")
def db_session(db_engine):
    Base.metadata.create_all(bind=db_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()
    # Po każdym teście czyścimy wszystko
    Base.metadata.drop_all(bind=db_engine)