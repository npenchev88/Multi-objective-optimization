# Multi-Objective Portfolio Optimization

This project explores multi-objective optimization techniques applied to stock portfolio selection, comparing evolutionary algorithms against random baselines.

## Experiment Overview

The experiment involves fetching historical financial data, processing it to derive key performance metrics, and solving a constrained multi-objective knapsack-style problem.

### 1. Data Acquisition & Processing
- **Source:** NASDAQ-listed symbols fetched via `yfinance`.
- **Period:** `2025-01-01` to `2025-12-31` (Daily interval).
- **Filtering:** Tickers with missing or incomplete data are automatically excluded.
- **Metrics Calculated:**
    - **Expected Return (Value):** Calculated as `(Latest Price / First Price) - 1`.
    - **Expected Risk:** Variance of returns over the period.
    - **Cost (Weight):** The latest available price per share.
    - **Market Cap:** Used for stock grading.

### 2. Stock Grading Logic
Stocks are categorized into four grades based on their Market Cap percentiles to enforce diversification constraints:
- **Grade A:** Top 10% (Largest Market Cap).
- **Grade B:** 70th to 90th percentile.
- **Grade C:** 40th to 70th percentile.
- **Grade D:** Bottom 40% (Smallest Market Cap).

### 3. Optimization Problem Definition
The problem is modeled as a **Bi-Objective Constrained Knapsack Problem**:
- **Objective 1:** Maximize total portfolio return.
- **Objective 2:** Minimize total portfolio risk.
- **Primary Constraint:** Total portfolio cost must not exceed **$10,000**.
- **Grade Constraints (Capacity Allocation):**
    - **Grade A:** Max 40% of total capacity ($4,000).
    - **Grade B:** Max 30% of total capacity ($3,000).
    - **Grade C:** Max 20% of total capacity ($2,000).
    - **Grade D:** Max 10% of total capacity ($1,000).

### 4. Experimental Setup
The study compares two primary methods across different problem scales, highlighting the difference between constrained evolutionary search and unconstrained random selection:

- **Methods:**
    - **NSGA2 (Constrained):**
        - **Algorithm:** Non-dominated Sorting Genetic Algorithm II.
        - **Constraints:** Strictly enforces total capacity ($10k) and grade-specific limits via a custom `PortfolioRepair` operator and feasible initialization.
        - **Search:** Evolutionary search for the Pareto front (maximizing return, minimizing risk).
    - **RANDOM (Unconstrained Baseline):**
        - **Strategy:** Selects exactly 30 unique assets at random.
        - **Constraints:** **Ignores** total capacity and grade-specific limits.
        - **Purpose:** Serves as a naive benchmark to compare against the optimized, constrained portfolios.
- **Scales (Problem Sizes):** 884, 1768, and 2653 stocks.
- **Replication:** 30 independent runs (seeds 0-29) for each configuration.
- **Evolutionary Parameters (NSGA2):**
    - Population Size: 300.
    - Generations: 200 (Total 60,000 evaluations per run).

## Project Structure
- `data/`: Scripts for fetching, normalizing, and grading stock data.
- `multiobj/`: Core optimization logic, problem definitions, and main execution script.
- `report/`: Metrics calculation (HV, IGD) and visualization tools.
- `multiobj_outputs/`: Raw results from optimization runs.
- `report/out/`: Generated reports, plots, and summary metrics.
