
import numpy as np
from pymoo.core.problem import ElementwiseProblem

def load_inputs(n_size=None):
    """Load weights, values, risk from knapsack_inputs and compute capacity dynamically."""
    from knapsack_solvers.data import knapsack_inputs as K

    def pick(obj, *names):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    weights = pick(K, "weights", "WEIGHTS")
    values  = pick(K, "values",  "VALUES")
    risk    = pick(K, "risk",    "RISK")

    if weights is None or values is None or risk is None:
        raise ValueError("knapsack_inputs.py must define weights, values, and risk arrays")

    W = np.asarray(weights, dtype=float).ravel()
    V = np.asarray(values,  dtype=float).ravel()
    R = np.asarray(risk,    dtype=float).ravel()

    N_full = len(W)
    if not (len(V) == len(R) == N_full):
        raise ValueError("weights, values, risk must all be the same length")

    if n_size is None:
        n_size = N_full
    if n_size > N_full:
        raise ValueError(f"Requested N={n_size}, but dataset length is only {N_full}")

    W_cut, V_cut, R_cut = W[:n_size], V[:n_size], R[:n_size]
    capacity = int(W_cut.sum() / 2)   # <-- compute here

    return W_cut, V_cut, R_cut, capacity


class KnapsackBiObjective(ElementwiseProblem):
    """Bi-objective knapsack-like problem: maximize value, minimize risk, subject to capacity."""
    def __init__(self, N):
        W, V, R, capacity = load_inputs(N)
        self.W, self.V, self.R, self.capacity = W, V, R, capacity
        super().__init__(n_var=N, n_obj=2, n_constr=1, xl=0, xu=1, type_var=np.bool_)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(int)
        total_w = float(self.W @ x)
        total_v = float(self.V @ x)
        total_r = float(self.R @ x)

        f1 = -total_v   # maximize value
        f2 = total_r    # minimize risk
        g1 = total_w - self.capacity  # <= 0 is feasible

        out["F"] = np.array([f1, f2])
        out["G"] = np.array([g1])
