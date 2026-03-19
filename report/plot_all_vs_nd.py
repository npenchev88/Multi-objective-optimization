import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set backend for headless environments
import matplotlib
matplotlib.use('Agg')

OUT_DIR = "multiobj_outputs"
REPORT_OUT = "report/out"

def plot_all_vs_nd(n_size=36, seed=1):
    os.makedirs(REPORT_OUT, exist_ok=True)
    
    methods = ["NSGA2", "RANDOM"]
    colors = {"NSGA2": "#1f77b4", "RANDOM": "#ff7f0e"}
    
    # 1. Plot ALL solutions together
    plt.figure(figsize=(10, 7))
    for method in methods:
        all_file = os.path.join(OUT_DIR, "all_solutions", f"all_2obj_{method}_N{n_size}_seed{seed}.csv")
        if os.path.exists(all_file):
            df = pd.read_csv(all_file)
            plt.scatter(df['f2'], -df['f1'], s=5, alpha=0.3, label=f"{method} (All)", color=colors[method])
    
    plt.title(f"All Evaluated Solutions (N={n_size}, Seed={seed})")
    plt.xlabel("Risk (f2)")
    plt.ylabel("Value (-f1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    all_img_path = os.path.join(REPORT_OUT, f"all_solutions_N{n_size}.png")
    plt.savefig(all_img_path, dpi=150)
    print(f"Saved all solutions plot to {all_img_path}")
    plt.close()

    # 2. Plot ONLY Non-Dominated solutions together
    plt.figure(figsize=(10, 7))
    for method in methods:
        front_file = os.path.join(OUT_DIR, "fronts", f"front_2obj_{method}_N{n_size}_seed{seed}.csv")
        if os.path.exists(front_file):
            df = pd.read_csv(front_file)
            plt.scatter(df['f2'], -df['f1'], s=20, alpha=0.9, label=f"{method} (ND)", color=colors[method], edgecolors='k')
            
            # Optionally add a line for the front
            df_sorted = df.sort_values(by='f2')
            plt.plot(df_sorted['f2'], -df_sorted['f1'], color=colors[method], alpha=0.5, linestyle='--')

    plt.title(f"Non-Dominated (ND) Solutions (N={n_size}, Seed={seed})")
    plt.xlabel("Risk (f2)")
    plt.ylabel("Value (-f1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    nd_img_path = os.path.join(REPORT_OUT, f"nd_solutions_N{n_size}.png")
    plt.savefig(nd_img_path, dpi=150)
    print(f"Saved ND solutions plot to {nd_img_path}")
    plt.close()

if __name__ == "__main__":
    plot_all_vs_nd(n_size=36, seed=1)
