# app/components/data_cleaning.py

import pandas as pd
from typing import Optional


class BudgetDataCleaner:
    """
    Class responsible for loading and preparing budget data.
    """

    def __init__(self, filepath: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None):
        """
        Initializes the BudgetDataCleaner class.

        Args:
            filepath (str, optional): Path to the CSV file.
            dataframe (pd.DataFrame, optional): A ready DataFrame to load.
        """
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

        if dataframe is not None:
            self.df = dataframe.copy()
        elif filepath is not None:
            self.load_data()

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        if self.filepath is None:
            raise ValueError("No filepath provided to load data.")
        
        try:
            self.df = pd.read_csv(self.filepath, sep=",")
            return self.df
        except Exception as e:
            raise ValueError(f"Error while loading data from {self.filepath}: {e}")

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans and prepares the dataset:
        - Removes spaces from column names
        - Converts the 'Date' column to datetime format
        - Normalizes values in 'Type' and 'Payment Method' columns
        - Ensures the 'Amount' column is of float type
        - Creates additional 'Year' and 'Month' columns for filtering

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        if self.df is None:
            raise ValueError("Data has not been loaded yet. Please use load_data() first.")

        df = self.df.copy()

        df.columns = df.columns.str.strip()

        category_map = {
            'entertainment': 'Entertainment',
            'enntertainment': 'Entertainment'
        }

        payment_method_map = {
            'cassh': 'Cash',
            'cash': 'Cash',
            'caasg': 'Cash'
        }

        df['Category'] = df['Category'].str.lower().replace(category_map).str.capitalize()
        df['Payment Method'] = df['Payment Method'].str.lower().replace(payment_method_map).str.capitalize()

        if 'Date' in df.columns:
            if df['Date'].dtype == object:
                df['Date'] = df['Date'].str.replace('/', '-', regex=False)
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Date'] = df['Date'].dt.date

        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

        df.dropna(inplace=True)

        self.df = df
        return self.df

    def get_clean_dataframe(self) -> pd.DataFrame:
        """
        Getter for the cleaned dataset.

        Returns:
            pd.DataFrame: A cleaned and ready-to-use DataFrame.
        
        Raises:
            ValueError: If the data has not been loaded and cleaned yet.
        """
        if self.df is None:
            raise ValueError("Data has not been loaded and cleaned yet.")
        return self.df

