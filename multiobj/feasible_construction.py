import numpy as np

def construct_feasible_random_solution(W, Grades, capacity, grade_limits, rng):
    """
    Constructs a random feasible solution incrementally.
    Uses a random target number of items to ensure diversity when constraints are loose.
    """
    N = len(W)
    solution = np.zeros(N, dtype=np.uint8)
    indices = np.arange(N)
    rng.shuffle(indices)
    
    # Randomly decide how many items we WANT to try to pick (up to N)
    # This ensures diversity even if all items fit in the knapsack.
    if hasattr(rng, 'integers'):
        n_to_pick = rng.integers(1, N + 1)
    else:
        n_to_pick = rng.randint(1, N + 1)
    indices_to_try = indices[:n_to_pick]
    
    current_total_weight = 0.0
    current_grade_weights = {g: 0.0 for g in grade_limits}
    
    limits = {g: pct * capacity for g, pct in grade_limits.items()}
    
    for idx in indices_to_try:
        item_weight = W[idx]
        item_grade = Grades[idx]
        
        # Check total capacity
        if current_total_weight + item_weight > capacity:
            continue
            
        # Check grade capacity if item has a grade with a limit
        if item_grade in limits:
            if current_grade_weights[item_grade] + item_weight > limits[item_grade]:
                continue
        
        # Add item
        solution[idx] = 1
        current_total_weight += item_weight
        if item_grade in current_grade_weights:
            current_grade_weights[item_grade] += item_weight
            
    return solution

def repair_solution(solution, W, Grades, capacity, grade_limits, rng):
    """
    Repairs an infeasible solution by removing items randomly until it becomes feasible.
    """
    N = len(W)
    sol = solution.copy().astype(np.uint8)
    limits = {g: pct * capacity for g, pct in grade_limits.items()}
    
    # Check and repair grade limits first
    for g, limit in limits.items():
        mask = (Grades == g)
        grade_indices = np.where(sol & mask)[0]
        grade_weight = np.dot(W[mask], sol[mask])
        
        while grade_weight > limit and grade_indices.size > 0:
            remove_idx = rng.choice(grade_indices)
            sol[remove_idx] = 0
            # Re-calculate to avoid issues with changed indices
            grade_indices = np.where(sol & mask)[0]
            grade_weight = np.dot(W[mask], sol[mask])
            
    # Check and repair total capacity
    total_weight = np.dot(W, sol)
    while total_weight > capacity:
        item_indices = np.where(sol == 1)[0]
        if not item_indices.size:
            break
        remove_idx = rng.choice(item_indices)
        sol[remove_idx] = 0
        total_weight = np.dot(W, sol)
        
    return sol
