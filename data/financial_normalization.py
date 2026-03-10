import pandas as pd
import os
import yfinance as yf

def normalize_financial_data(input_file='data/raw_sample_stocks.csv'):
    """
    Reads raw stock data and calculates expected return, risk, cost, and market cap.
    Drops any tickers (columns) that have missing or no data.

    Expected Return = (Latest Price / First Price) - 1
    Expected Risk = Variance of returns for the period
    Cost = Latest Price
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return None

    df = pd.read_csv(input_file)

    for col in ['Date', 'Datetime', 'Unnamed: 0']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Remove all tickers with any missing data
    df_clean = df.dropna(axis=1, how='any')

    dropped_count = len(df.columns) - len(df_clean.columns)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} tickers due to missing or incomplete data.")

    if df_clean.empty:
        print("Error: No tickers remain after dropping missing data.")
        return None

    results = []

    for ticker in df_clean.columns:
        prices = df_clean[ticker]

        if len(prices) < 2:
            print(f"Skipping {ticker}: not enough price points.")
            continue

        first_price = prices.iloc[0]
        latest_price = prices.iloc[-1]

        if pd.isna(first_price) or pd.isna(latest_price) or first_price == 0:
            print(f"Skipping {ticker}: invalid first/latest price.")
            continue

        print(f"Checking metadata for {ticker}")
        ticker_obj = yf.Ticker(ticker)


        # example. pct_change calculates the percentage change (current / previous) - 1
        # s = pd.Series([100, 110, 105])
        # print(s.pct_change())
        # 0         NaN
        # 1    0.100000
        # 2   -0.045455
        # dtype: float64
        # That's why we need to drop the first one because it will be NaN
        returns = prices.pct_change().dropna()
        expected_return = returns.mean()
        expected_risk = returns.var()

        cost = latest_price

        # Market cap from fast_info
        try:
            fi = ticker_obj.fast_info
            market_cap = fi.get('marketCap', 0) or 0
        except Exception as e:
            print(f"Skipping {ticker}: fast_info failed: {e}")
            continue

        if market_cap == 0:
            print(f"Skipping {ticker}: market cap is 0 or missing.")
            continue

        if expected_risk == 0:
            print(f"Skipping {ticker}: expected_risk is 0")
            continue

        results.append({
            'Ticker': ticker,
            'Expected Return': expected_return,
            'Expected Risk (Var)': expected_risk,
            'Cost (Latest Price)': cost,
            'Market Cap': market_cap
        })

    normalized_df = pd.DataFrame(results)

    return normalized_df.sort_values("Ticker").reset_index(drop=True)


if __name__ == "__main__":
    normalized_data = normalize_financial_data()

    if normalized_data is not None:
        print(f"\nNormalized Financial Data for {len(normalized_data)} tickers:")
        print(normalized_data.head(10).to_string(index=False))
        if len(normalized_data) > 10:
            print("...")

        output_path = os.path.join('data', 'normalized_stocks.csv')
        normalized_data.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\nNormalized data saved to {output_path}")