import pandas as pd
import numpy as np
import glob
import os
import re

# --- CONFIGURATION ---
ALL_SOLUTIONS_DIR = "multiobj_outputs/all_solutions"
BASE_OUT = "report/out"
OUTPUT_DIR = os.path.join(BASE_OUT, "split_all_solutions")

COL_RETURN = "f1 (Return)"
COL_RISK = "f2 (Risk)"
# For dense populations, the ratio might be smaller than for sparse fronts.
# We'll use 5.0 as a safe default, but the logic now runs per file.
GAP_THRESHOLD = 5.0 

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_metadata(filename):
    """Extracts method, N, and seed from filename."""
    basename = os.path.basename(filename)
    match = re.search(r"all_2obj_(?P<method>[^_]+)_N(?P<N>\d+)_seed(?P<seed>\d+)", basename)
    if match:
        return match.group("method"), match.group("N"), match.group("seed")
    return None, None, None

def split_by_max_gap(df):
    """
    Finds the largest Risk gap in the dataframe and splits it.
    Returns: (left_df, right_df, did_split)
    """
    if len(df) < 2:
        return df, df.iloc[:0], False
    
    # 1. Sort by Risk
    df_sorted = df.sort_values(by=COL_RISK).reset_index(drop=True)
    risks = df_sorted[COL_RISK].values
    
    # 2. Compute gaps
    gaps = np.diff(risks)
    if len(gaps) == 0:
        return df_sorted, df_sorted.iloc[:0], False
        
    max_idx = np.argmax(gaps)
    max_gap = gaps[max_idx]
    median_gap = np.median(gaps)
    
    # 3. Apply Safety Rule
    ratio = max_gap / median_gap if median_gap > 0 else 0
    
    if ratio >= GAP_THRESHOLD:
        left = df_sorted.iloc[:max_idx + 1]
        right = df_sorted.iloc[max_idx + 1:]
        return left, right, True
    else:
        return df_sorted, df_sorted.iloc[:0], False

def main():
    setup_directories()
    files = glob.glob(os.path.join(ALL_SOLUTIONS_DIR, "all_2obj_*.csv"))
    
    if not files:
        print(f"No files found in {ALL_SOLUTIONS_DIR}")
        return

    processed_count = 0
    split_count = 0

    print(f"Processing {len(files)} all_solutions files...")

    for f in files:
        try:
            method, N, seed = extract_metadata(f)
            if method is None:
                print(f"Skipping {f}: Could not extract metadata.")
                continue
            
            # Read only necessary columns first to check for presence and speed up
            df = pd.read_csv(f)
            if COL_RISK not in df.columns:
                print(f"Skipping {f}: Missing {COL_RISK} column.")
                continue
            
            # Remove exact duplicates to ensure gaps are calculated on unique points
            df = df.drop_duplicates()

            # Find gap and split directly on this file's data
            left, right, did_split = split_by_max_gap(df)
            
            base = os.path.basename(f).replace(".csv", "")
            
            # Save files
            left.to_csv(os.path.join(OUTPUT_DIR, f"{base}_LEFT.csv"), index=False)
            right.to_csv(os.path.join(OUTPUT_DIR, f"{base}_RIGHT.csv"), index=False)
            
            processed_count += 1
            if did_split:
                split_count += 1
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("\n" + "="*40)
    print("ALL SOLUTIONS SPLIT COMPLETE")
    print(f"Files processed: {processed_count}")
    print(f"Files split:     {split_count}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()
