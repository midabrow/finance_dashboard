-- Create table: Expenses
CREATE TABLE IF NOT EXISTS expenses (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL, 
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    type VARCHAR(20) CHECK (type IN ('Income', 'Expense')) NOT NULL, 
    amount NUMERIC(12, 2) NOT NULL CHECK(amount >= 0),
    payment_method VARCHAR(50),
    status VARCHAR(20) CHECK (status IN ('Completed', 'Pending')) DEFAULT 'Completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table: Investments
CREATE TABLE IF NOT EXISTS investments (
    id SERIAL PRIMARY KEY,
    purchase_date DATE NOT NULL,
    company_name TEXT NOT NULL,
    ticker_symbol VARCHAR(10) NOT NULL,
    shares INTEGER NOT NULL CHECK (shares > 0),
    purchase_price_usd NUMERIC(12, 2) NOT NULL CHECK (purchase_price_usd >= 0),
    account_type VARCHAR(20) CHECK (account_type IN ('Standard', 'IKE', 'IKZE')) DEFAULT 'Standard',
    currency VARCHAR(5) DEFAULT 'USD',
    status VARCHAR(20) CHECK (status IN ('Active', 'Sold')) DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);

-- Index for filtering/sorting expenses by date
CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);

-- Index for filtering/sorting investments by ticker
CREATE INDEX IF NOT EXISTS idx_investments_ticker ON investments(ticker_symbol);