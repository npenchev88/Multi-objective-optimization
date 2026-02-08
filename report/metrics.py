import pandas as pd
import numpy as np
import glob
import json
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import os
os.environ["MPLBACKEND"] = "Agg"

# --- NEW: helpers for thinning ---
def _thin_by_grid(points: np.ndarray, grid: float = 1e-3, max_points: int = 5000) -> np.ndarray:
    """
    Агресивно разреждане: снэпваме към решетка (grid) и пазим по 1 точка на клетка.
    След това, ако пак са много, избираме равномерно разпределени индекси.
    """
    if points.size == 0:
        return points
    f1 = np.round(points[:, 0] / grid).astype(np.int64)
    f2 = np.round(points[:, 1] / grid).astype(np.int64)
    key = f1 * 1_000_003 + f2  # примитивен хеш
    _, idx = np.unique(key, return_index=True)
    thinned = points[np.sort(idx)]
    if thinned.shape[0] > max_points:
        # равномерно семплиране по индекси (по-евтино от k-means/farthest)
        sel = np.linspace(0, thinned.shape[0] - 1, num=max_points, dtype=int)
        thinned = thinned[sel]
    return thinned

def _non_dominated(P: np.ndarray) -> np.ndarray:
    if P.size == 0:
        return P
    idx = NonDominatedSorting().do(P, only_non_dominated_front=True)
    return P[idx]

def load_fronts(fronts_dir):
    """
    Чете само нужните колони с ниска точност (float32) и категорична колона за method,
    и прави ранно разреждане per (N, method, seed), за да не расте паметта.
    """
    files = glob.glob(f"{fronts_dir}/*.csv")
    if not files:
        return pd.DataFrame(columns=["N","method","seed","f1","f2"])

    dfs = []
    usecols = ["N","method","seed","f1","f2"]
    dtypes = {"N":"int32","method":"category","seed":"int32","f1":"float32","f2":"float32"}
    for f in files:
        df = pd.read_csv(f, usecols=usecols, dtype=dtypes)
        df = df.dropna(subset=["f1","f2"])
        # Ранно ND + thinning per run:
        parts = []
        for (_, g) in df.groupby(["N","method","seed"], observed=True):
            P = g[["f1","f2"]].to_numpy(dtype=np.float32)
            nd = _non_dominated(P)
            thin = _thin_by_grid(nd, grid=1e-3, max_points=2000)  # <= 2000 точки/рун
            gg = pd.DataFrame(thin, columns=["f1","f2"], dtype=np.float32)
            # възстанови груповите ключове
            for col in ["N","method","seed"]:
                gg[col] = g.iloc[0][col]
            parts.append(gg[["N","method","seed","f1","f2"]])
        if parts:
            dfs.append(pd.concat(parts, ignore_index=True))
    if not dfs:
        return pd.DataFrame(columns=["N","method","seed","f1","f2"])
    return pd.concat(dfs, ignore_index=True)
# def load_fronts(fronts_dir):
#     """Reads all front CSVs and concatenates them."""
#     files = glob.glob(f"{fronts_dir}/*.csv")
#     if not files:
#         return pd.DataFrame()
#     df_list = [pd.read_csv(f) for f in files]
#     return pd.concat(df_list, ignore_index=True)

def load_meta(logs_dir):
    """Reads all JSON meta."""
    files = glob.glob(f"{logs_dir}/*.json")
    if not files:
        return pd.DataFrame()
    meta_list = []
    for f in files:
        with open(f, 'r') as file:
            meta_list.append(json.load(file))
    return pd.DataFrame(meta_list)

def non_dominated(F):
    """Returns the non-dominated set from F."""
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]

def pooled_reference_front(df_fronts, N):
    """Pools all methods & seeds at size N and returns the non-dominated set."""
    df_n = df_fronts[df_fronts['N'] == N]
    if df_n.empty:
        return np.array([])
    pooled_front = df_n[['f1', 'f2']].values
    return non_dominated(pooled_front)

def hv_igd_tables(df_fronts):
    """Computes HV, IGD+, and |ND| for each run."""
    results = []
    if df_fronts.empty:
        return pd.DataFrame(columns=['N', 'method', 'seed', 'HV', 'IGD', 'ND_size'])

    for n_val in df_fronts['N'].unique():
        ref_front = pooled_reference_front(df_fronts, n_val)
        if ref_front.shape[0] == 0:
            continue

        max_vals = np.max(ref_front, axis=0)
        ref_point = max_vals * 1.1
        
        igd_metric = IGDPlus(ref_front)
        hv_metric = HV(ref_point=ref_point)

        for (method, seed), group in df_fronts[df_fronts['N'] == n_val].groupby(['method', 'seed']):
            front = group[['f1', 'f2']].values
            nd_front = non_dominated(front)
            
            if nd_front.shape[0] > 0:
                hv = hv_metric.do(nd_front)
                igd = igd_metric.do(nd_front)
                nd_size = len(nd_front)
            else:
                hv = 0
                igd = np.inf
                nd_size = 0

            results.append({
                'N': n_val,
                'method': method,
                'seed': seed,
                'HV': hv,
                'IGD': igd,
                'ND_size': nd_size
            })
            
    return pd.DataFrame(results)

def aggregate_ci(df, by=["N", "method"]):
    """Computes mean and 95% CI."""
    
    def ci(series):
        n = len(series)
        if n == 0:
            return 0
        mean = series.mean()
        std = series.std()
        ci_val = 1.96 * std / np.sqrt(n)
        return ci_val

    agg_funcs = {
        'HV': ['mean', ci],
        'IGD': ['mean', ci],
        'ND_size': ['mean', ci]
    }
    
    summary = df.groupby(by).agg(agg_funcs).reset_index()
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    for metric in ['HV', 'IGD', 'ND_size']:
        summary[f'{metric}_ci_low'] = summary[f'{metric}_mean'] - summary[f'{metric}_ci']
        summary[f'{metric}_ci_high'] = summary[f'{metric}_mean'] + summary[f'{metric}_ci']
        summary = summary.drop(columns=[f'{metric}_ci'])

    return summary.rename(columns={'N_': 'N', 'method_': 'method'})
