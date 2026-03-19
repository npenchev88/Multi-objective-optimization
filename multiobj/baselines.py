import time
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def random_feasible_evalboxed(N, W, V, R, Grades, Tickers, capacity, grade_limits, n_evals, seed, batch_size=2048):
    """
    Generates solutions by selecting exactly 30 random assets (non-duplicates) 
    for a fixed number of evaluations.
    Ignores capacity and grade limits as per user request.
    Returns all evaluated solutions and objectives.
    """
    rng = np.random.default_rng(seed)
    n_evaluated = 0

    all_solutions = []
    all_objectives = []

    # Number of assets to select in each portfolio
    K = 30
    # Ensure K does not exceed N
    if N < K:
        K = N

    while n_evaluated < n_evals:
        current_batch_size = min(batch_size, n_evals - n_evaluated)
        solutions = np.zeros((current_batch_size, N), dtype=np.uint8)
        objectives = np.empty((current_batch_size, 2), dtype=np.float32)

        for i in range(current_batch_size):
            # Choose 30 unique indices
            indices = rng.choice(N, size=K, replace=False)
            sol = np.zeros(N, dtype=np.uint8)
            sol[indices] = 1
            
            solutions[i] = sol
            
            # [-Value, Risk]
            objectives[i, 0] = -np.dot(V, sol)
            objectives[i, 1] = np.dot(R, sol)

        all_solutions.append(solutions)
        all_objectives.append(objectives)
        n_evaluated += current_batch_size

    return np.vstack(all_solutions), np.vstack(all_objectives)

def random_feasible_baseline(N, W, V, R, Grades, Tickers, capacity, grade_limits, n_samples, seed):
    """
    Generates random solutions by selecting exactly 30 unique assets, 
    removes duplicates based on objective values, and returns the non-dominated front.
    """
    rng = np.random.default_rng(seed)

    # Number of assets to select in each portfolio
    K = 30
    if N < K:
        K = N

    feasible_solutions = []
    for _ in range(n_samples):
        indices = rng.choice(N, size=K, replace=False)
        sol = np.zeros(N, dtype=np.uint8)
        sol[indices] = 1
        feasible_solutions.append(sol)

    feasible_solutions = np.array(feasible_solutions, dtype=np.uint8)
    
    # Calculate objectives: [-Value, Risk]
    objectives = np.array([
        [-np.dot(V, sol), np.dot(R, sol)] for sol in feasible_solutions
    ], dtype=np.float32)

    # Remove duplicate solutions based on objective vectors for efficiency
    unique_objectives, unique_indices = np.unique(
        objectives, axis=0, return_index=True
    )
    unique_solutions = feasible_solutions[unique_indices]

    # Get the non-dominated front from the unique set
    nd_indices = NonDominatedSorting().do(unique_objectives, only_non_dominated_front=True)

    # Return the final non-dominated solutions and their objectives
    return unique_solutions[nd_indices], unique_objectives[nd_indices]
