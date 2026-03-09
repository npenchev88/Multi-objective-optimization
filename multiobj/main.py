import os
import json
import time
import random
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from multiobj.problem_biobj import KnapsackMultiObjective, load_inputs
from multiobj.baselines import random_feasible_evalboxed
from multiobj.operators import FeasibleRandomSampling, PortfolioRepair
METHODS = ["NSGA2", "RANDOM"]
POP_SIZE = 300
N_GEN = 200
N_EVALS = POP_SIZE * N_GEN

SIZES = [884, 1768, 2653]
SEEDS = range(0, 30)
CAPACITY = 10000.0

OUT_DIR = "multiobj_outputs"

# Grade-based constraints (fraction of total capacity)
GRADE_LIMITS = {
    "A": 0.4,
    "B": 0.3,
    "C": 0.2,
    "D": 0.1
}

def run_all():
    """Main function to run all bi-objective experiments."""
    os.makedirs(os.path.join(OUT_DIR, "fronts"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "all_solutions"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)

    for n_size_req in SIZES:
        try:
            W, V, R, Grades, Tickers, capacity = load_inputs(n_size_req, capacity=CAPACITY)
            # Use actual number of stocks loaded
            actual_n = len(W)
        except Exception as e:
            print(f"Skipping N={n_size_req}: {e}")
            continue

        for seed in SEEDS:
            # Set global seeds for overall consistency
            random.seed(seed)
            np.random.seed(seed)

            for method in METHODS:
                print(f"Running N={actual_n}, Seed={seed}, Method={method}, Capacity={CAPACITY}")

                start_time = time.perf_counter()

                if method == "NSGA2":
                    problem = KnapsackMultiObjective(N=actual_n, grade_limits=GRADE_LIMITS, capacity=CAPACITY)

                    algorithm = NSGA2(
                        pop_size=POP_SIZE,
                        sampling=FeasibleRandomSampling(),
                        crossover=TwoPointCrossover(),
                        mutation=BitflipMutation(),
                        repair=PortfolioRepair(),
                        eliminate_duplicates=True
                    )
                    termination = get_termination("n_gen", N_GEN)
                    
                    res = minimize(problem, algorithm, termination, seed=seed, verbose=False, save_history=True)
                    
                    final_X, final_F = res.X, res.F
                    
                    all_X_list = []
                    all_F_list = []
                    for gen in res.history:
                        all_X_list.append(gen.pop.get("X"))
                        all_F_list.append(gen.pop.get("F"))
                    
                    all_X = np.vstack(all_X_list)
                    all_F = np.vstack(all_F_list)

                elif method == "RANDOM":
                    # Get 2obj results from baseline a
                    all_X, all_F = random_feasible_evalboxed(
                        actual_n, W, V, R, Grades, Tickers, capacity, GRADE_LIMITS, n_evals=N_EVALS, seed=seed
                    )
                    
                    # all_costs = np.array([float(np.dot(W, x)) for x in raw_all_X], dtype=np.float32).reshape(-1, 1)
                    # all_X = raw_all_X
                    # all_F = np.hstack([raw_all_F_2obj, all_costs])
                    
                    nd_indices = NonDominatedSorting().do(all_F, only_non_dominated_front=True)
                    final_X = all_X[nd_indices]
                    final_F = all_F[nd_indices]

                elapsed_s = time.perf_counter() - start_time

                def get_ticker_strings(X_arr, Tickers_arr):
                    res_list = []
                    for sol in X_arr:
                        selected_indices = np.where(sol == 1)[0]
                        selected_tickers = Tickers_arr[selected_indices]
                        res_list.append(",".join(selected_tickers))
                    return res_list

                def get_extra_metrics(X_arr, W_arr, Grades_arr, cap):
                    metrics = []
                    grades_unique = sorted(list(GRADE_LIMITS.keys()))
                    for sol in X_arr:
                        row = {}
                        for g in grades_unique:
                            mask = (Grades_arr == g)
                            w_g = float(np.dot(W_arr[mask], sol[mask]))
                            row[f"grade_{g}_pct"] = round(w_g / cap, 2)
                        metrics.append(row)
                    return pd.DataFrame(metrics)

                def reorder_columns(df):
                    cols = [c for c in df.columns if c != 'tickers'] + ['tickers']
                    return df[cols]

                # Pareto Front Data
                front_df = pd.DataFrame(final_F, columns=["f1 (Return)", "f2 (Risk)"])
                front_df["tickers"] = get_ticker_strings(final_X, Tickers)
                extra_front = get_extra_metrics(final_X, W, Grades, capacity)
                front_df = pd.concat([front_df, extra_front], axis=1)
                
                front_df["method"] = method
                front_df["N"] = actual_n
                front_df["seed"] = seed
                front_df = reorder_columns(front_df)
                
                front_filename = f"front_2obj_{method}_N{actual_n}_seed{seed}.csv"
                front_df.to_csv(os.path.join(OUT_DIR, "fronts", front_filename), index=False)

                # ALL Explored Solutions Data
                all_df = pd.DataFrame(all_F, columns=["f1 (Return)", "f2 (Risk)"])
                all_df["tickers"] = get_ticker_strings(all_X, Tickers)
                extra_all = get_extra_metrics(all_X, W, Grades, capacity)
                all_df = pd.concat([all_df, extra_all], axis=1)

                all_df["method"] = method
                all_df["N"] = actual_n
                all_df["seed"] = seed
                all_df = reorder_columns(all_df)
                
                all_filename = f"all_2obj_{method}_N{actual_n}_seed{seed}.csv"
                all_df.to_csv(os.path.join(OUT_DIR, "all_solutions", all_filename), index=False)

                meta = {
                    "N": actual_n,
                    "method": method,
                    "seed": seed,
                    "n_evals": N_EVALS,
                    "elapsed_s": elapsed_s,
                    "capacity": capacity,
                    "pop_size": POP_SIZE if method == "NSGA2" else None,
                    "n_gen": N_GEN if method == "NSGA2" else None,
                    "nd_count": len(front_df),
                    "all_count": len(all_df),
                }
                meta_filename = f"meta_2obj_{method}_N{actual_n}_seed{seed}.json"
                with open(os.path.join(OUT_DIR, "logs", meta_filename), "w") as f:
                    json.dump(meta, f, indent=4)

if __name__ == "__main__":
    run_all()
