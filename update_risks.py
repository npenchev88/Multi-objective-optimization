
import pandas as pd
import numpy as np
import os

def update_risks():
    raw_file = 'data/old/raw_sample_stocks.csv'
    graded_file = 'data/old/graded_stocks.csv'
    
    if not os.path.exists(raw_file) or not os.path.exists(graded_file):
        print("Missing data files.")
        return

    raw_df = pd.read_csv(raw_file)
    graded_df = pd.read_csv(graded_file)
    
    # Drop non-ticker columns from raw
    for col in ['Date', 'Datetime', 'Unnamed: 0']:
        if col in raw_df.columns:
            raw_df = raw_df.drop(columns=[col])
            
    # Calculate pct_change variance for each ticker in graded_df
    new_risks = []
    for ticker in graded_df['Ticker']:
        if ticker in raw_df.columns:
            prices = raw_df[ticker].dropna()
            if len(prices) >= 2:
                risk = prices.pct_change().var()
                new_risks.append(risk)
            else:
                new_risks.append(0.0)
        else:
            new_risks.append(0.0)
            
    graded_df['Expected Risk (Var)'] = new_risks
    graded_df.to_csv(graded_file, index=False, float_format='%.4f')
    print(f"Updated risks in {graded_file} using percentage return variance.")

if __name__ == "__main__":
    update_risks()
