
import numpy as np
from pymoo.core.problem import ElementwiseProblem

def load_inputs(n_size=None, capacity=10000.0):
    """Load weights, values, risk, grades, and tickers from data/graded_stocks.csv."""
    import pandas as pd
    import os

    # Use absolute path or relative to project root
    file_path = os.path.join('data', 'graded_stocks.csv')
    if not os.path.exists(file_path):
        # Try one level up if called from inside multiobj/
        file_path = os.path.join('..', 'data', 'graded_stocks.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")

    df = pd.read_csv(file_path)

    N_full = len(df)
    if n_size is None or n_size > N_full:
        n_size = N_full

    # Get n_size random positions from the population once (using fixed seed 42)
    if n_size < N_full:
        df = df.sample(n=n_size, random_state=42).reset_index(drop=True)

    # Values = Expected Return, Risk = Expected Risk (Var), Weight = Cost (Latest Price)
    V_cut = df['Expected Return'].values
    R_cut = df['Expected Risk (Var)'].values
    W_cut = df['Cost (Latest Price)'].values
    G_cut = df['Grade'].values
    T_cut = df['Ticker'].values

    return W_cut, V_cut, R_cut, G_cut, T_cut, capacity


class KnapsackMultiObjective(ElementwiseProblem):
    """
    Multi-objective knapsack-like problem: 
    1. maximize value (return)
    2. minimize risk
    Subject to capacity and grade constraints.
    """
    def __init__(self, N, grade_limits=None, capacity=10000.0):
        W, V, R, Grades, Tickers, capacity = load_inputs(N, capacity=capacity)
        self.W, self.V, self.R, self.Grades, self.Tickers, self.capacity = W, V, R, Grades, Tickers, capacity


        self.grade_limits = grade_limits or {}
        
        # Actual size might be smaller than requested N if N > N_full
        actual_N = len(self.W)
        
        # Pre-calculate masks for each grade for faster evaluation
        self.masks = {g: (self.Grades == g) for g in self.grade_limits}
        
        # Number of constraints: 1 (total capacity) + number of grade limits
        n_constr = 1 + len(self.grade_limits)

        super().__init__(n_var=actual_N, n_obj=2, n_constr=n_constr, xl=0, xu=1, type_var=np.bool_)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(int)
        total_w = float(self.W @ x)
        total_v = float(self.V @ x)
        total_r = float(self.R @ x)

        f1 = -total_v   # maximize return
        f2 = total_r    # minimize risk
        
        # Constraints: g <= 0 is feasible
        constraints = [total_w - self.capacity]
        
        for g, limit_pct in self.grade_limits.items():
            mask = self.masks[g]
            w_g = float(self.W[mask] @ x[mask])
            constraints.append(w_g - limit_pct * self.capacity)

        out["F"] = np.array([f1, f2])
        out["G"] = np.array(constraints)
