import numpy as np
import pandas as pd
import os
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Mock the problem and construction
def construct_feasible_random_solution(W, Grades, capacity, grade_limits, rng):
    N = len(W)
    solution = np.zeros(N, dtype=np.uint8)
    indices = np.arange(N)
    rng.shuffle(indices)
    
    n_to_pick = rng.integers(1, N + 1)
    indices_to_try = indices[:n_to_pick]
    
    current_total_weight = 0.0
    current_grade_weights = {g: 0.0 for g in grade_limits}
    limits = {g: pct * capacity for g, pct in grade_limits.items()}
    
    for idx in indices_to_try:
        item_weight = W[idx]
        item_grade = Grades[idx]
        if current_total_weight + item_weight > capacity:
            continue
        if item_grade in limits:
            if current_grade_weights[item_grade] + item_weight > limits[item_grade]:
                continue
        solution[idx] = 1
        current_total_weight += item_weight
        if item_grade in current_grade_weights:
            current_grade_weights[item_grade] += item_weight
    return solution

def run_repro(N=3500, n_evals=40000, seed=1):
    rng = np.random.default_rng(seed)
    
    # Generate dummy data
    # Prices around 100-500
    W = rng.uniform(100, 500, N)
    # Returns 0.01 to 0.15
    V = rng.uniform(0.01, 0.15, N)
    # Risk 0.01 to 0.10, positively correlated with Return
    R = V * 0.5 + rng.uniform(0, 0.05, N)
    
    Grades = rng.choice(["A", "B", "C", "D"], N)
    capacity = 10000.0
    grade_limits = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    
    all_F = np.empty((n_evals, 2))
    
    print(f"Generating {n_evals} random solutions for N={N}...")
    for i in range(n_evals):
        sol = construct_feasible_random_solution(W, Grades, capacity, grade_limits, rng)
        all_F[i, 0] = -np.dot(V, sol) # Minimize -Return
        all_F[i, 1] = np.dot(R, sol)  # Minimize Risk
        
    nd_indices = NonDominatedSorting().do(all_F, only_non_dominated_front=True)
    print(f"Number of non-dominated solutions: {len(nd_indices)}")
    
    if len(nd_indices) > 0:
        print("Sample non-dominated solutions (f1, f2):")
        print(all_F[nd_indices[:5]])

if __name__ == "__main__":
    run_repro()
