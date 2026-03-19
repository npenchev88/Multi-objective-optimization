import pandas as pd
import os

def apply_grade_logic():
    file_path = os.path.join('data', 'normalized_stocks.csv')
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Read the data
    df = pd.read_csv(file_path)

    # Fill NaN in Market Cap with 0
    df['Market Cap'] = df['Market Cap'].fillna(0)

    # Calculate percentiles (ranking from largest to smallest)
    # Using rank(pct=True) where 1.0 is the largest
    df['market_cap_rank'] = df['Market Cap'].rank(pct=True)

    # Logic:
    # First 10% (top 90% to 100%) -> A
    # Next 20% (70% to 90%) -> B
    # Next 30% (40% to 70%) -> C
    # Remaining 40% (0% to 40%) -> D

    def assign_grade(rank):
        if rank > 0.9:
            return 'A'
        elif rank > 0.7:
            return 'B'
        elif rank > 0.4:
            return 'C'
        else:
            return 'D'

    df['Grade'] = df['market_cap_rank'].apply(assign_grade)

    # Drop the temporary rank column
    df = df.drop(columns=['market_cap_rank'])

    # Save the result to a new file or overwrite? 
    # Usually safer to create a new file first or just overwrite if specified.
    # The prompt says "create another file ... where we'll read", 
    # so I'll save the output to data/graded_stocks.csv and print a summary.
    output_path = os.path.join('data', 'graded_stocks.csv')
    df.to_csv(output_path, index=False)

    print(f"Successfully applied grade logic. Saved to {output_path}")
    print("\nGrade Distribution:")
    print(df['Grade'].value_counts(normalize=True) * 100)
    print("\nFirst 5 rows:")
    print(df[['Ticker', 'Market Cap', 'Grade']].head())

if __name__ == "__main__":
    apply_grade_logic()
