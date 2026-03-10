import os
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib to avoid Qt warnings

import pandas as pd
from datetime import datetime

from metrics import (
    load_fronts, load_meta, hv_igd_tables, aggregate_ci
)
from plots import (
    plot_optimization_process, plot_box_multi, plot_runtime,
    plot_population_only, plot_pareto_only, plot_metric_box,
    plot_pareto_per_seed
)

FRONTS_DIR = "multiobj_outputs/fronts"
LOGS_DIR = "multiobj_outputs/logs"
ALL_SOLUTIONS_DIR = "multiobj_outputs/all_solutions"
OUTPUT_DIR = "report/out"

def main():
    """Main script to generate the report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df_fronts = load_fronts(FRONTS_DIR)
    meta_df = load_meta(LOGS_DIR)

    if df_fronts.empty or meta_df.empty:
        report_content = "# Experiment Report\n\n**Warning: No data found in `multiobj_outputs`. Report generation skipped.**"
        with open(os.path.join(OUTPUT_DIR, "REPORT.md"), "w") as f:
            f.write(report_content)
        print("No data found. Generated a placeholder report.")
        return

    # Compute metrics
    metrics_per_run = hv_igd_tables(df_fronts)
    metrics_agg = aggregate_ci(metrics_per_run)

    # Save metric CSVs
    metrics_per_run.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics_per_run.csv"), index=False)
    metrics_agg.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics_agg.csv"), index=False)

    # Generate plots
    # Ensure N is treated as numeric for sorting
    df_fronts['N'] = pd.to_numeric(df_fronts['N'])
    unique_Ns = sorted(df_fronts['N'].unique())
    
    # Process Plots for different N
    for n in unique_Ns:
        # Combined multi-panel
        plot_optimization_process(ALL_SOLUTIONS_DIR, df_fronts, n, os.path.join(OUTPUT_DIR, f"process_aggregated_N{n}.png"))
        
        # Individual panels for paper
        plot_population_only(ALL_SOLUTIONS_DIR, n, os.path.join(OUTPUT_DIR, f"population_all_N{n}.png"))
        plot_pareto_only(df_fronts, n, os.path.join(OUTPUT_DIR, f"pareto_all_N{n}.png"))

        # Combined multi-panel boxplots
        plot_box_multi(metrics_per_run, n, os.path.join(OUTPUT_DIR, f"box_multi_N{n}.png"))
        
        # Individual metric boxplots
        plot_metric_box(metrics_per_run, n, 'HV', 'HV Distribution', 'Hypervolume', os.path.join(OUTPUT_DIR, f"HV_box_N{n}.png"))
        plot_metric_box(metrics_per_run, n, 'IGD', 'IGD Distribution', 'IGD+', os.path.join(OUTPUT_DIR, f"IGD_box_N{n}.png"))
        plot_metric_box(metrics_per_run, n, 'ND_size', '|ND| Distribution', 'Number of ND points', os.path.join(OUTPUT_DIR, f"ND_size_box_N{n}.png"))

        # # Per-seed Pareto Fronts. For testing purposes
        # seeds_for_n = sorted(df_fronts[df_fronts['N'] == n]['seed'].unique())
        # for s in seeds_for_n:
        #     plot_pareto_per_seed(df_fronts, n, s, os.path.join(OUTPUT_DIR, f"pareto_N{n}_seed{s}.png"))

    plot_runtime(meta_df, os.path.join(OUTPUT_DIR, "runtime.png"))

    # Generate Markdown Report
    generate_markdown_report(meta_df, metrics_agg, unique_Ns, df_fronts)


def generate_markdown_report(meta_df, metrics_agg, unique_Ns, df_fronts):
    """Generates the final Markdown report file."""
    # Lists of Ns and methods actually present in the metrics table
    Ns = sorted(pd.to_numeric(metrics_agg['N']).unique().tolist())
    methods = sorted(metrics_agg['method'].unique().tolist())

    def render_metric_table(metric: str, is_int: bool = False) -> str:
        # Build a Markdown table as pure strings (no numeric DF to avoid dtype issues)
        header = "| N | " + " | ".join(methods) + " |\n"
        sep = "|" + "|".join(["---"] * (len(methods) + 1)) + "|\n"
        lines = [header, sep]

        for N in Ns:
            row_cells = []
            for m in methods:
                row = metrics_agg[(pd.to_numeric(metrics_agg["N"]) == N) & (metrics_agg["method"] == m)]
                if row.empty:
                    cell = "—"
                else:
                    mean = row[f"{metric}_mean"].iloc[0]
                    lo   = row[f"{metric}_ci_low"].iloc[0]
                    hi   = row[f"{metric}_ci_high"].iloc[0]
                    if pd.isna(mean) or pd.isna(lo) or pd.isna(hi):
                        cell = "—"
                    else:
                        if is_int:
                            cell = f"{mean:.0f} ({lo:.0f} - {hi:.0f})"
                        else:
                            cell = f"{mean:.3f} ({lo:.3f} - {hi:.3f})"
                row_cells.append(cell)
            lines.append("| " + str(N) + " | " + " | ".join(row_cells) + " |\n")

        return "".join(lines)

    report_content = f"""# Multi-Objective Knapsack-like Portfolio Optimization using Evolutionary Algorithms
**Authors:** Nikolay Penchev and Angel Marchev Jr.

## Introduction to the Problem
In finance, portfolio optimization aims to balance risk and return. For large portfolios with many assets, traditional methods (e.g., exhaustive search) become computationally expensive.
This report presents an experimental comparison of algorithms for a multi-objective knapsack-like portfolio optimization problem. The goal is to select a portfolio of assets that simultaneously maximizes the expected 'Value' (return) and minimizes the associated 'Risk'. These two objectives are often in conflict, requiring a trade-off. Multi-objective optimization is well-suited for this problem as it does not require a single, arbitrary weighting of risk versus return, but instead identifies a set of optimal trade-off solutions, known as the Pareto front.

## What is the Knapsack Problem?
A classic combinatorial optimization problem
- You have a knapsack with a capacity W.
- There are N items, each with value and weight
- The Goal is to Maximize total value without exceeding the knapsack capacity. 

<img src="knapsack.png" alt="Knapsack" width="600"/>


In a portfolio context:

- Capacity -> budget/limit

- Item -> asset (e.g., stock, bond, etc.)

- Item(Weight) -> investment cost

- Item(Value) -> expected return

- Item(Risk) -> Volatility/Uncertainty

<img src="portfolio.png" alt="Portfolio" width="600"/>


## Evolutionary Algorithms (EA)
- Key Idea: Inspired by natural selection.
- Main Steps:
  - Initialization of a population (random solutions).
  - Evaluation (Fitness) of each solution.
  - Selection of the best solutions.
  - Recombination (Crossover) and Mutation to create new solutions.
  - Repeat until stopping criteria are met.

<img src="darwin_2.png" alt="Darwin" width="600"/>


## Algorithms

Two algorithms were compared in this study:

- **NSGA-II (Non-dominated Sorting Genetic Algorithm II):** A widely-used evolutionary algorithm for multi-objective optimization. It employs mechanisms of selection, crossover, and mutation to iteratively evolve a population of solutions toward the true Pareto front. Its key features include a fast non-dominated sorting procedure and a crowding distance mechanism to maintain diversity among solutions.

- **Random Search:** This method serves as a baseline for comparison. It generates solutions randomly within the search space. In this experiment, it is given a fixed time budget or generation count comparable to NSGA-II to evaluate its efficiency.

## Optimization Process Visualization

To understand how these algorithms work, we visualize the process of evaluating solutions and identifying the best trade-offs across all random seeds. The left panel shows the search space explored, while the right panel shows the combined Pareto front found by each algorithm.

"""
    # Show the optimization process for each N
    for n in unique_Ns:
        report_content += f"### Optimization Process and Pareto Front for N={n}\n"
        report_content += f"![Optimization Process N={n}](process_aggregated_N{n}.png)\n\n"

    report_content += f"""## Data Description

We generate asset data using the np.random.normal function from the NumPy library. For each asset, three prop-
erties were generated: investment cost (weight), return, and
risk. The weights were drawn from a normal distribution
with a mean of 10 and a standard deviation of 3.

## 1. Setup

This report summarizes the performance of multi-objective optimization methods.

- **Problem Sizes (N):** {unique_Ns}
- **Methods:** {methods}
- **Seeds:** {sorted(df_fronts['seed'].unique().tolist())}
- **Total Runs:** {len(meta_df)}

**Objective Interpretation:**
- **Value(Expected return):** `-f1` (higher is better)
- **Risk:** `f2` (lower is better)

The goal is to find solutions that maximize the Expected return while minimizing Risk, representing a classic Pareto trade-off.

## 2. Performance Metrics

Metrics are aggregated across seeds (mean ± 95% CI).

- **HV (Hypervolume) ↑:** Measures the volume of the dominated portion of the objective space. Higher is better.
- **IGD+ (Inverted Generational Distance Plus) ↓:** Measures the average distance from each point in the reference front to the obtained front. Lower is better.
- **|ND| (Number of Non-Dominated Points) ↑:** The number of points in the final Pareto front. Higher is generally better, indicating more choices.

### HV (↑) mean ± 95% CI

"""

    report_content += render_metric_table("HV", is_int=False)
    report_content += "\n### IGD+ (↓) mean ± 95% CI\n\n"
    report_content += render_metric_table("IGD", is_int=False)
    report_content += "\n### |ND| (↑) mean ± 95% CI\n\n"
    report_content += render_metric_table("ND_size", is_int=True)

    report_content += """\n## 3. Combined Pareto Fronts for Different N

Scatter plots of **Risk vs. Value**.  
As N increases, the problem complexity grows. These plots show the union of non-dominated solutions across all seeds.

"""

    for n in unique_Ns:
        report_content += f"### N = {n}\n![Pareto Front for N={n}](pareto_N{n}.png)\n"

    report_content += """\n## 4. Performance Distribution (Boxplots)

"""
    for n in unique_Ns:
        report_content += f"### N = {n}\n![N={n} Combined](box_multi_N{n}.png)\n"


    report_content += """\n## 5. Runtime Overview

One of the key observations is the speed of the **Random Search** method. Since it does not perform complex sorting or evolutionary operations, it is significantly faster than NSGA-II. In our experiments, Random Search is used as a baseline to see if the extra computational cost of NSGA-II is justified by the quality of the Pareto front it finds.

![Runtime Overview](runtime.png)

"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "REPORT.md"), "w") as f:
        f.write(report_content)



if __name__ == "__main__":
    main()
    print(f"Report generated in {OUTPUT_DIR}/")
