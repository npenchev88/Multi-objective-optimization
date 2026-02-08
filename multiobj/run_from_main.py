import os
import json
import time
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from multiobj.problem_biobj import KnapsackBiObjective, load_inputs
from multiobj.baselines import random_feasible_timeboxed

METHODS = ["NSGA2", "RANDOM"]
POP_SIZE = 200
SIZES = [10, 100, 1000, 10000]
TIME_CAPS = {10: 25, 100: 100, 1000: 175, 10000: 250}
SEEDS = list(range(30))
OUT_DIR = "multiobj_outputs"
def run_all():
    """Main function to run all bi-objective experiments."""
    os.makedirs(os.path.join(OUT_DIR, "fronts"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)

    for n_size in SIZES:
        for seed in SEEDS:
            for method in METHODS:
                print(f"Running N={n_size}, Seed={seed}, Method={method}")
                time_cap_s = TIME_CAPS[n_size]
                start_time = time.perf_counter()

                try:
                    W, V, R, capacity = load_inputs(n_size)
                except ValueError as e:
                    print(f"Skipping N={n_size}: {e}")
                    continue

                if method == "NSGA2":
                    problem = KnapsackBiObjective(N=n_size)
                    algorithm = NSGA2(
                        pop_size=POP_SIZE,
                        sampling=BinaryRandomSampling(),
                        crossover=TwoPointCrossover(),
                        mutation=BitflipMutation(),
                        eliminate_duplicates=True
                    )
                    termination = get_termination("time", time_cap_s)
                    res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
                    final_X, final_F = res.X, res.F

                elif method == "RANDOM":
                    final_X, final_F, n_samples_timeboxed = random_feasible_timeboxed(
                        n_size, W, V, R, capacity, time_cap_s=time_cap_s, seed=seed, batch_size=2048
                    )

                elapsed_s = time.perf_counter() - start_time

                # Ensure we have results to save
                if final_X is None or final_F is None or len(final_F) == 0:
                    print("No results to save.")
                    continue

                # Save Pareto front
                front_df = pd.DataFrame(final_F, columns=["f1", "f2"])
                front_df["method"] = method
                front_df["N"] = n_size
                front_df["seed"] = seed
                front_df["selected_count"] = [np.sum(x) for x in final_X]
                front_filename = f"front_2obj_{method}_N{n_size}_seed{seed}.csv"
                front_df.to_csv(os.path.join(OUT_DIR, "fronts", front_filename), index=False)

                # Save metadata
                meta = {
                    "N": n_size,
                    "method": method,
                    "seed": seed,
                    "time_cap_s": time_cap_s,
                    "elapsed_s": elapsed_s,
                    "pop_size": POP_SIZE if method == "NSGA2" else None,
                    "saved_rows": len(front_df),
                }
                if method == "NSGA2":
                    meta["n_evals"] = res.algorithm.evaluator.n_eval
                elif method == "RANDOM":
                    meta["n_samples_timeboxed"] = n_samples_timeboxed
                meta_filename = f"meta_2obj_{method}_N{n_size}_seed{seed}.json"
                with open(os.path.join(OUT_DIR, "logs", meta_filename), "w") as f:
                    json.dump(meta, f, indent=4)