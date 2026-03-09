# Data Management for Multi-objective Optimization

This directory contains utilities for fetching and processing financial data to be used in portfolio optimization problems.

## Scripts

### 1. `stock_data_fetcher.py`
This script uses the `yfinance` library to download historical stock prices for a large set of tickers.

**Usage:**
```bash
python3 data/stock_data_fetcher.py
```

**Key Features:**
- **Ticker Source:** Reads tickers from `data/nasdaq-listed/nasdaq-listed.csv`.
- **Filtering:** Automatically filters for tickers where `market == 'us_market'` and `marketCap > 0` using individual Ticker info calls.
- **Batching:** Downloads data in batches (e.g., 100-200 tickers) to optimize performance and handle large datasets.
- **Output:** Saves raw historical closing prices to `data/raw_sample_stocks.csv`.

---

### 2. `financial_normalization.py`
This script processes the raw stock data to generate metrics suitable for multi-objective optimization.

**Usage:**
```bash
python3 data/financial_normalization.py
```

**Key Features:**
- **Data Cleaning:** Automatically drops any ticker (column) that has any missing data (NaNs) during the period to ensure data integrity for optimization.
- **Calculated Metrics:**
    - **Expected Return:** Calculated as `(Latest Price / First Price) - 1`.
    - **Expected Risk:** Calculated as the **Variance** of the stock prices over the selected period.
    - **Cost:** Defined as the **Latest Price** of the stock in the period.

**Outputs:**
- Saves the processed metrics to `data/normalized_stocks.csv`.

---

### 3. `apply_grade_logic.py`
This script classifies stocks into market cap-based grades to be used as an additional constraint or objective in the optimization process.

**Usage:**
```bash
python3 data/apply_grade_logic.py
```

**Key Features:**
- **Data Source:** Reads from `data/normalized_stocks.csv`.
- **Preprocessing:** Fills missing `Market Cap` values with 0.
- **Grading Logic:**
    - **Grade A:** Top 10% of stocks by Market Cap.
    - **Grade B:** Next 20% of stocks.
    - **Grade C:** Next 30% of stocks.
    - **Grade D:** Remaining 40% of stocks.

**Outputs:**
- Saves the graded stocks to `data/graded_stocks.csv`.

---

## Data Files
- `nasdaq-listed/nasdaq-listed.csv`: Source list of NASDAQ tickers.
- `raw_sample_stocks.csv`: Historical closing prices for the filtered tickers.
- `normalized_stocks.csv`: Processed metrics (Return, Risk, Cost) for each ticker.
- `graded_stocks.csv`: Final dataset including Return, Risk, Cost, Market Cap, and Grade.
