import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# Set backend for headless environments
os.environ["MPLBACKEND"] = "Agg"

# --- CONFIGURATION ---
INPUT_DIR = "multiobj_outputs/fronts"
INPUT_PATTERN = os.path.join(INPUT_DIR, "front_2obj_*.csv")
BASE_OUT = "report/out"
OUTPUT_DIRS = {
    "split": os.path.join(BASE_OUT, "split_fronts"),
    "agg": os.path.join(BASE_OUT, "aggregate_fronts"),
    "plots": os.path.join(BASE_OUT, "plots_split")
}
COL_RETURN = "f1 (Return)"
COL_RISK = "f2 (Risk)"
GAP_THRESHOLD = 5.0

def setup_directories():
    for d in OUTPUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

def extract_metadata(filename):
    """Extracts method, N, and seed from filename."""
    basename = os.path.basename(filename)
    match = re.search(r"front_2obj_(?P<method>[^_]+)_N(?P<N>\d+)_seed(?P<seed>\d+)", basename)
    if match:
        return match.group("method"), match.group("N"), match.group("seed")
    return "Unknown", "Unknown", "Unknown"

def get_non_dominated_2d(df):
    """Returns non-dominated points for 2D minimization (f1 and f2)."""
    if df.empty:
        return df
    
    # Sort: primary f1 (asc), secondary f2 (asc)
    pts_df = df.sort_values(by=[COL_RETURN, COL_RISK])
    pts = pts_df.values
    
    f1_idx = pts_df.columns.get_loc(COL_RETURN)
    f2_idx = pts_df.columns.get_loc(COL_RISK)
    
    non_dominated = []
    min_f2 = float('inf')
    
    for row in pts:
        if row[f2_idx] < min_f2:
            non_dominated.append(row)
            min_f2 = row[f2_idx]
            
    return pd.DataFrame(non_dominated, columns=df.columns)

def split_by_gap(df):
    """Splits the front based on the largest Risk (f2) gap."""
    if len(df) < 2:
        return df, df.iloc[:0], {"max_gap": 0, "median_gap": 0, "ratio": 0, "split": False, "idx": 0}
    
    df_sorted = df.sort_values(by=COL_RISK).reset_index(drop=True)
    risks = df_sorted[COL_RISK].values
    
    gaps = np.diff(risks)
    max_gap = np.max(gaps)
    median_gap = np.median(gaps)
    max_idx = np.argmax(gaps)
    
    ratio = max_gap / median_gap if median_gap > 0 else 0
    
    if ratio >= GAP_THRESHOLD:
        left = df_sorted.iloc[:max_idx + 1]
        right = df_sorted.iloc[max_idx + 1:]
        return left, right, {"max_gap": max_gap, "median_gap": median_gap, "ratio": ratio, "split": True, "idx": max_idx}
    else:
        return df_sorted, df_sorted.iloc[:0], {"max_gap": max_gap, "median_gap": median_gap, "ratio": ratio, "split": False, "idx": 0}

def save_plot(df_left, df_right, title, filename, is_aggregate=False):
    """Generates scatter plots for fronts without connecting lines across gaps."""
    plt.figure(figsize=(10, 7))
    
    # Apply paper-style formatting
    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    label_font = {'fontsize': 14, 'fontweight': 'bold'}
    
    if not df_left.empty:
        plt.scatter(df_left[COL_RISK], -df_left[COL_RETURN], c='#1f77b4', label='Left Component', s=25, alpha=0.7)
        if is_aggregate:
            sorted_l = df_left.sort_values(by=COL_RISK)
            plt.plot(sorted_l[COL_RISK], -sorted_l[COL_RETURN], color='#1f77b4', linestyle='--', alpha=0.4)

    if not df_right.empty:
        plt.scatter(df_right[COL_RISK], -df_right[COL_RETURN], c='#ff7f0e', label='Right Component', s=25, alpha=0.7)
        if is_aggregate:
            sorted_r = df_right.sort_values(by=COL_RISK)
            plt.plot(sorted_r[COL_RISK], -sorted_r[COL_RETURN], color='#ff7f0e', linestyle='--', alpha=0.4)

    plt.title(title, fontdict=title_font)
    plt.xlabel("Expected Risk", fontdict=label_font)
    plt.ylabel("Expected Return", fontdict=label_font)
    plt.grid(True, alpha=0.3)
    plt.legend(prop={'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRS["plots"], filename), dpi=200)
    plt.close()

def main():
    setup_directories()
    files = glob.glob(INPUT_PATTERN)
    
    if not files:
        print(f"No files found matching {INPUT_PATTERN}")
        return

    summary_rows = []
    # Nested pools: method -> N -> side -> list of dataframes
    pools = {} 
    
    processed_count = 0
    split_count = 0

    print(f"Processing {len(files)} files...")

    for f in files:
        try:
            df = pd.read_csv(f)
            if COL_RETURN not in df.columns or COL_RISK not in df.columns:
                print(f"Skipping {f}: Missing expected columns.")
                continue
            
            method, N, seed = extract_metadata(f)
            left, right, m = split_by_gap(df)
            
            base = os.path.basename(f).replace(".csv", "")
            left.to_csv(os.path.join(OUTPUT_DIRS["split"], f"{base}_LEFT.csv"), index=False)
            right.to_csv(os.path.join(OUTPUT_DIRS["split"], f"{base}_RIGHT.csv"), index=False)
            
            save_plot(left, right, f"Split Front: {method} (N={N}, Seed={seed})", f"{base}_split.png")
            
            # Organize pools by method AND N
            if method not in pools: pools[method] = {}
            if N not in pools[method]: pools[method][N] = {"left": [], "right": []}
            
            pools[method][N]["left"].append(left)
            pools[method][N]["right"].append(right)
            
            summary_rows.append({
                "file_name": base, "method": method, "N": N, "seed": seed,
                "total_points": len(df), "left_points": len(left), "right_points": len(right),
                "max_gap": m["max_gap"], "median_gap": m["median_gap"],
                "gap_ratio": m["ratio"], "did_split": m["split"]
            })
            
            processed_count += 1
            if m["split"]: split_count += 1
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Create Aggregates per Method and N
    print("Generating aggregate unions...")
    stats_lines = []
    for method in pools:
        for N in pools[method]:
            for side in ["left", "right"]:
                side_upper = side.upper()
                all_pts_list = pools[method][N][side]
                
                if not any(not d.empty for d in all_pts_list):
                    continue
                
                # 1. All Points
                df_all = pd.concat(all_pts_list, ignore_index=True)
                all_filename = f"all_{side_upper}_{method}_N{N}"
                df_all.to_csv(os.path.join(OUTPUT_DIRS["agg"], f"{all_filename}.csv"), index=False)
                save_plot(df_all if side == "left" else pd.DataFrame(), 
                          df_all if side == "right" else pd.DataFrame(), 
                          f"All Points: {method} {side_upper} (N={N})", f"{all_filename}.png")
                
                # 2. Union Pareto
                df_union = get_non_dominated_2d(df_all)
                union_filename = f"union_{side_upper}_{method}_N{N}"
                df_union.to_csv(os.path.join(OUTPUT_DIRS["agg"], f"{union_filename}.csv"), index=False)
                save_plot(df_union if side == "left" else pd.DataFrame(), 
                          df_union if side == "right" else pd.DataFrame(), 
                          f"Union Pareto: {method} {side_upper} (N={N})", f"{union_filename}.png", is_aggregate=True)
                
                stats_lines.append(f"{method} N={N} {side_upper}: All={len(df_all)}, Union={len(df_union)}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(BASE_OUT, "split_summary.csv"), index=False)

    print("\n" + "="*40)
    print("PROCESSING REPORT")
    print(f"Files processed: {processed_count}")
    print(f"Files split:     {split_count}")
    print("-" * 40)
    for line in stats_lines:
        print(line)
    print("="*40)
    print(f"Results saved in: {BASE_OUT}")

if __name__ == "__main__":
    main()
