import time
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def random_feasible_timeboxed(N, W, V, R, capacity, time_cap_s, seed, batch_size=2048, max_archive_size=10000):
    """
    Generates and repairs random solutions until time cap, maintaining only the
    non-dominated front to conserve memory.
    """
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    n_samples_evaluated = 0

    # Archive for non-dominated solutions and their objectives
    nd_solutions = np.empty((0, N), dtype=np.uint8)
    nd_objectives = np.empty((0, 2), dtype=np.float32)

    while time.perf_counter() - start_time < time_cap_s:
        # Generate a batch of solutions
        solutions = rng.integers(2, size=(batch_size, N), dtype=np.uint8)
        n_samples_evaluated += batch_size

        # Repair and evaluate objectives for the batch
        objectives = np.empty((batch_size, 2), dtype=np.float32)
        for i, sol in enumerate(solutions):
            weight = np.dot(W, sol)
            while weight > capacity:
                item_indices = np.where(sol == 1)[0]
                if not item_indices.size:
                    break
                remove_idx = rng.choice(item_indices)
                sol[remove_idx] = 0
                weight = np.dot(W, sol)
            
            objectives[i, 0] = -np.dot(V, sol)
            objectives[i, 1] = np.dot(R, sol)

        # Combine new results with the current non-dominated front
        combined_solutions = np.vstack([nd_solutions, solutions])
        combined_objectives = np.vstack([nd_objectives, objectives])

        # Find the new non-dominated front
        nd_indices = NonDominatedSorting().do(combined_objectives, only_non_dominated_front=True)
        
        nd_solutions = combined_solutions[nd_indices]
        nd_objectives = combined_objectives[nd_indices]

        # Optional: Thin the archive if it grows too large
        if len(nd_solutions) > max_archive_size:
            # Simple thinning: randomly sample a subset
            indices_to_keep = rng.choice(len(nd_solutions), size=max_archive_size, replace=False)
            nd_solutions = nd_solutions[indices_to_keep]
            nd_objectives = nd_objectives[indices_to_keep]

    return nd_solutions, nd_objectives, n_samples_evaluated

def random_feasible_baseline(N, W, V, R, capacity, n_samples, seed):
    """
    Generates random solutions, repairs them, removes duplicates based on 
    objective values, and returns the non-dominated front.

    Args:
        N (int): Number of items.
        W (np.array): Weights of items.
        V (np.array): Values of items.
        R (np.array): Risks of items.
        capacity (float): Knapsack capacity.
        n_samples (int): Number of random solutions to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - np.array: The non-dominated solutions.
            - np.array: The corresponding objective values for the non-dominated solutions.
    """
    rng = np.random.default_rng(seed)

    # 1. Limit n_samples to the total size of the search space (2^N)
    # This is crucial for small N to avoid redundant sampling.
    if N < 60:  # Avoid overflow for large N
        actual_samples = min(n_samples, 2**N)
    else:
        actual_samples = n_samples
        
    solutions = rng.integers(2, size=(actual_samples, N), dtype=np.uint8)
    feasible_solutions = []

    # Repair solutions to be feasible
    for sol in solutions:
        weight = np.dot(W, sol)
        while weight > capacity:
            item_indices = np.where(sol == 1)[0]
            if not item_indices.size:
                break
            remove_idx = rng.choice(item_indices)
            sol[remove_idx] = 0
            weight = np.dot(W, sol)
        feasible_solutions.append(sol)

    feasible_solutions = np.array(feasible_solutions, dtype=np.uint8)
    
    # Calculate objectives: [-Value, Risk]
    objectives = np.array([
        [-np.dot(V, sol), np.dot(R, sol)] for sol in feasible_solutions
    ], dtype=np.float32)

    # 2. Remove duplicate solutions based on objective vectors for efficiency
    # Using np.unique is much faster than pandas for large arrays.
    # It returns the indices of the first occurrence of each unique row.
    unique_objectives, unique_indices = np.unique(
        objectives, axis=0, return_index=True
    )
    unique_solutions = feasible_solutions[unique_indices]

    # 3. Get the non-dominated front from the unique set
    nd_indices = NonDominatedSorting().do(unique_objectives, only_non_dominated_front=True)

    # Return the final non-dominated solutions and their objectives
    return unique_solutions[nd_indices], unique_objectives[nd_indices]
