import yfinance as yf
import pandas as pd
import os


def get_nasdaq_tickers(take=100, file_path='data/nasdaq-listed/nasdaq-listed.csv'):
    """Reads tickers from the provided CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    df = pd.read_csv(file_path).sort_values("Ticker", ascending=True).head(take)
    return df['Ticker'].tolist()


def fetch_stock_data_batched(tickers, start_period='2025-01-01', end_period='2025-12-31', interval='1d',
                             batch_size=100):
    """
    Fetches historical stock data in batches and concatenates them.
    """
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1}: {batch[:3]}... ({len(batch)} tickers)")

        try:
            data = yf.download(batch, start=start_period, end=end_period, interval=interval, group_by='ticker',
                               threads=True)

            if data.empty:
                print(f"Batch {i // batch_size + 1} returned empty data.")
                continue

            batch_close = pd.DataFrame()
            for t in batch:
                if t in data.columns.levels[0]:
                    batch_close[t] = data[t]['Close']

            all_data.append(batch_close)

        except Exception as e:
            print(f"Error downloading batch: {e}")

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, axis=1)
    return final_df


if __name__ == "__main__":
    START = '2026-02-01'
    END = '2026-03-01'
    INTERVAL = '1d'
    BATCH_SIZE = 200
    LIMIT_FOR_FILTERING = None
    TAKE_N = 4200

    nasdaq_tickers = get_nasdaq_tickers(take=TAKE_N)

    if nasdaq_tickers:
        df = fetch_stock_data_batched(nasdaq_tickers, start_period=START, end_period=END, interval=INTERVAL,
                                      batch_size=BATCH_SIZE)

        if not df.empty:
            print("\nFirst 5 rows of final data:")
            print(df.head())

            output_path = os.path.join('data', 'raw_sample_stocks.csv')
            df.to_csv(output_path, float_format='%.4f')
            print(f"\nData saved to {output_path}")
        else:
            print("No data fetched.")
    else:
        print("No valid tickers found to download.")
