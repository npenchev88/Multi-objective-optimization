# Multi-Objective Knapsack-like Portfolio Optimization using Evolutionary Algorithms
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

### Optimization Process and Pareto Front for N=884
![Optimization Process N=884](process_aggregated_N884.png)

### Optimization Process and Pareto Front for N=1768
![Optimization Process N=1768](process_aggregated_N1768.png)

### Optimization Process and Pareto Front for N=2653
![Optimization Process N=2653](process_aggregated_N2653.png)

## Data Description

We generate asset data using the np.random.normal function from the NumPy library. For each asset, three prop-
erties were generated: investment cost (weight), return, and
risk. The weights were drawn from a normal distribution
with a mean of 10 and a standard deviation of 3.

## 1. Setup

This report summarizes the performance of multi-objective optimization methods.

- **Problem Sizes (N):** [884, 1768, 2653]
- **Methods:** ['NSGA2', 'RANDOM']
- **Seeds:** [0, 1, 2, 3]
- **Total Runs:** 180

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

| N | NSGA2 | RANDOM |
|---|---|---|
| 884 | 2.006 (1.989 - 2.023) | 0.391 (0.351 - 0.430) |
| 1768 | 6.218 (6.087 - 6.349) | 0.884 (0.848 - 0.920) |
| 2653 | 12.600 (12.219 - 12.981) | 1.410 (1.364 - 1.456) |

### IGD+ (↓) mean ± 95% CI

| N | NSGA2 | RANDOM |
|---|---|---|
| 884 | 0.005 (0.002 - 0.008) | 0.800 (0.772 - 0.827) |
| 1768 | 0.020 (0.009 - 0.032) | 1.338 (1.327 - 1.348) |
| 2653 | 0.019 (0.008 - 0.031) | 1.512 (1.495 - 1.528) |

### |ND| (↑) mean ± 95% CI

| N | NSGA2 | RANDOM |
|---|---|---|
| 884 | 235 (231 - 239) | 25 (21 - 29) |
| 1768 | 251 (245 - 257) | 25 (21 - 29) |
| 2653 | 262 (257 - 266) | 24 (18 - 29) |

## 3. Combined Pareto Fronts for Different N

Scatter plots of **Risk vs. Value**.  
As N increases, the problem complexity grows. These plots show the union of non-dominated solutions across all seeds.

### N = 884
![Pareto Front for N=884](pareto_N884.png)
### N = 1768
![Pareto Front for N=1768](pareto_N1768.png)
### N = 2653
![Pareto Front for N=2653](pareto_N2653.png)

## 4. Performance Distribution (Boxplots)

### N = 884
![N=884 Combined](box_multi_N884.png)
### N = 1768
![N=1768 Combined](box_multi_N1768.png)
### N = 2653
![N=2653 Combined](box_multi_N2653.png)

## 5. Runtime Overview

One of the key observations is the speed of the **Random Search** method. Since it does not perform complex sorting or evolutionary operations, it is significantly faster than NSGA-II. In our experiments, Random Search is used as a baseline to see if the extra computational cost of NSGA-II is justified by the quality of the Pareto front it finds.

![Runtime Overview](runtime.png)

