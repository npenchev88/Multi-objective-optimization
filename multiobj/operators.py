import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from multiobj.feasible_construction import construct_feasible_random_solution, repair_solution

class FeasibleRandomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # Access problem attributes
        W = problem.W
        Grades = problem.Grades
        capacity = problem.capacity
        grade_limits = problem.grade_limits
        
        # We rely on np.random.seed(seed) called in main.py
        # and pass 'np.random' as the source of randomness
        X = np.empty((n_samples, problem.n_var), dtype=np.bool_)
        for i in range(n_samples):
            # Pass np.random to use the global state
            X[i] = construct_feasible_random_solution(W, Grades, capacity, grade_limits, np.random)
            
        return X

class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        W = problem.W
        Grades = problem.Grades
        capacity = problem.capacity
        grade_limits = problem.grade_limits
        
        # Use global np.random state
        X_repaired = np.empty_like(X, dtype=np.bool_)
        for i in range(len(X)):
            X_repaired[i] = repair_solution(X[i], W, Grades, capacity, grade_limits, np.random)
            
        return X_repaired
