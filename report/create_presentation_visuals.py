import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metrics import (
    load_fronts, load_meta, hv_igd_tables, aggregate_ci
)
from plots import (
    plot_pareto, plot_hv_box, plot_runtime
)

FRONTS_DIR = "multiobj_outputs/fronts"
LOGS_DIR = "multiobj_outputs/logs"
OUTPUT_DIR = "report/out"


def generate_aggregate_plots(metrics_agg, output_dir):
    """Generates line plots for aggregated metrics with confidence intervals."""
    if metrics_agg.empty:
        print("Skipping aggregate plots: metrics_agg is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {"NSGA2": "blue", "RANDOM": "orange"}

    for metric in ["HV", "IGD", "ND_size"]:
        plt.figure(figsize=(12, 8))
        for method in metrics_agg['method'].unique():
            method_df = metrics_agg[metrics_agg['method'] == method].sort_values('N')
            plt.plot(method_df['N'], method_df[f'{metric}_mean'], marker='o', linestyle='-', label=method, color=colors.get(method))
            plt.fill_between(method_df['N'], method_df[f'{metric}_ci_low'], method_df[f'{metric}_ci_high'], alpha=0.2, color=colors.get(method))
        
        plt.xlabel("Problem Size (N)")
        ylabel = f"{metric.replace('_', ' ').title()} {'↑' if metric in ['HV', 'ND_size'] else '↓'}"
        plt.ylabel(ylabel)
        plt.title(f"Aggregate {metric.replace('_', ' ')} vs. Problem Size")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"agg_{metric}_vs_N.png"), dpi=300, bbox_inches="tight")
        plt.close()

def generate_boxplots(metrics_per_run, output_dir):
    """Generates boxplots for HV and IGD per N."""
    if metrics_per_run.empty:
        print("Skipping boxplots: metrics_per_run is empty.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for n_size in metrics_per_run['N'].unique():
        for metric in ["HV", "IGD"]:
            plt.figure(figsize=(10, 7))
            df_n = metrics_per_run[metrics_per_run['N'] == n_size]
            methods = df_n['method'].unique()
            data = [df_n[df_n['method'] == m][metric] for m in methods]
            
            plt.boxplot(data, labels=methods)
            plt.title(f"{metric} Distribution for N={n_size}")
            plt.ylabel(metric)
            plt.savefig(os.path.join(output_dir, f"{metric}_box_N{n_size}.png"), dpi=300)
            plt.close()

def generate_runtime_plot(meta_df, output_dir):
    """Generates a bar plot for mean runtime."""
    if meta_df.empty:
        print("Skipping runtime plot: meta_df is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    summary = meta_df.groupby(['N', 'method'])['elapsed_s'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 8))
    n_sizes = sorted(summary['N'].unique())
    methods = summary['method'].unique()
    x = np.arange(len(n_sizes))
    width = 0.35
    
    for i, method in enumerate(methods):
        method_data = summary[summary['method'] == method]
        plt.bar(x + i * width, method_data['mean'], width, yerr=method_data['std'], label=method, capsize=5)

    if 'time_cap_s' in meta_df.columns:
        time_cap = meta_df['time_cap_s'].max()
        plt.axhline(y=time_cap, color='r', linestyle='--', label=f'Time Cap ({time_cap}s)')

    plt.xlabel("Problem Size (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Mean Runtime by Problem Size and Method")
    plt.xticks(x + width / 2, n_sizes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_bars.png"), dpi=300)
    plt.close()

def generate_pareto_plots(df_fronts, output_dir):
    """Generates Pareto front plots for each N."""
    if df_fronts.empty:
        print("Skipping Pareto plots: df_fronts is empty.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    for n in sorted(df_fronts['N'].unique()):
        plot_pareto(df_fronts, n, os.path.join(output_dir, f"pareto_N{n}.png"))

def generate_markdown_table(metrics_agg):
    """Generates a Markdown table from the aggregated metrics."""
    if metrics_agg.empty:
        return "Aggregated metrics table is empty."

    # Pivot the table to have methods as columns
    pivot_hv = metrics_agg.pivot(index='N', columns='method', values=['HV_mean', 'HV_ci_low', 'HV_ci_high'])
    pivot_igd = metrics_agg.pivot(index='N', columns='method', values=['IGD_mean', 'IGD_ci_low', 'IGD_ci_high'])
    pivot_nd = metrics_agg.pivot(index='N', columns='method', values=['ND_size_mean', 'ND_size_ci_low', 'ND_size_ci_high'])

    def format_metric(pivot_df, metric_name, is_int=False):
        header = f"| N | {' | '.join(pivot_df.columns.levels[1])} |\n"
        sep = f"|---|{'|---' * len(pivot_df.columns.levels[1])}|\n"
        body = ""
        for n_size, row in pivot_df.iterrows():
            cells = [f"{n_size}"]
            for method in pivot_df.columns.levels[1]:
                mean = row[(f'{metric_name}_mean', method)]
                low = row[(f'{metric_name}_ci_low', method)]
                high = row[(f'{metric_name}_ci_high', method)]
                if is_int:
                    cells.append(f"{mean:.0f} ({low:.0f}–{high:.0f})")
                else:
                    cells.append(f"{mean:.3f} ({low:.3f}–{high:.3f})")
            body += f"| {' | '.join(cells)} |\n"
        title = f"### {metric_name.replace('_', ' ').title()}\n"
        return title + header + sep + body

    hv_table = format_metric(pivot_hv, 'HV')
    igd_table = format_metric(pivot_igd, 'IGD')
    nd_table = format_metric(pivot_nd, 'ND_size', is_int=True)

    return hv_table + "\n" + igd_table + "\n" + nd_table


if __name__ == "__main__":
    # Load data
    df_fronts = load_fronts(FRONTS_DIR)
    meta_df = load_meta(LOGS_DIR)
    
    # Compute metrics
    metrics_per_run = hv_igd_tables(df_fronts)
    metrics_agg = aggregate_ci(metrics_per_run)

    # Generate artifacts
    generate_aggregate_plots(metrics_agg, OUTPUT_DIR)
    generate_boxplots(metrics_per_run, OUTPUT_DIR)
    generate_runtime_plot(meta_df, OUTPUT_DIR)
    generate_pareto_plots(df_fronts, OUTPUT_DIR)
    
    markdown_table = generate_markdown_table(metrics_agg)
    
    print("--- Aggregated Metrics Summary ---")
    print(markdown_table)
    print(f"\nPlots saved to {OUTPUT_DIR}")