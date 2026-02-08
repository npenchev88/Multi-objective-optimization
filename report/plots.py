import os; os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def _to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _union_nd_front(points):
    """points: ndarray shape (k,2) with [f1,f2] (both to MINIMIZE)."""
    if points.size == 0:
        return points
    # drop duplicates using a rounded key (robust to tiny fp noise)
    dfp = pd.DataFrame(points, columns=["f1","f2"])
    key = (dfp["f1"].round(6)).astype(str) + "|" + (dfp["f2"].round(6)).astype(str)
    dfp = dfp.loc[~key.duplicated()].reset_index(drop=True)

    P = dfp[["f1","f2"]].to_numpy(dtype=float)
    if len(P) <= 2:
        return P

    nd_idx = NonDominatedSorting().do(P, only_non_dominated_front=True)
    return P[nd_idx]

def plot_pareto(df_fronts, N, out_path):
    """
    Plot Risk (f2) vs Value (-f1) for each method using the union
    non-dominated front across all seeds. If the resulting front is tiny
    (<= 2 pts), fall back to plotting all available points for that method.
    """
    plt.figure(figsize=(7, 5))

    df_n = df_fronts[df_fronts["N"] == N].copy()
    if df_n.empty:
        plt.title(f"Pareto Fronts for N={N} (No data)")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # normalize and ensure numeric
    df_n["method"] = df_n["method"].astype(str).str.strip().str.upper()
    df_n = _to_numeric(df_n, ["f1","f2"])

    colors  = {"NSGA2": "#1f77b4", "RANDOM": "#ff7f0e"}
    markers = {"NSGA2": "o",       "RANDOM": "^"}

    any_points = False
    for method, grp in df_n.groupby("method"):
        pts_all = grp[["f1","f2"]].dropna().to_numpy(dtype=float)

        # union ND front across all seeds
        nd = _union_nd_front(pts_all)

        # fallback: if ND is suspiciously tiny, show all unique points
        if nd.shape[0] <= 2 and pts_all.shape[0] > 2:
            # unique with rounding to avoid overplot
            dfu = pd.DataFrame(pts_all, columns=["f1","f2"])
            key = (dfu["f1"].round(6)).astype(str) + "|" + (dfu["f2"].round(6)).astype(str)
            dfu = dfu.loc[~key.duplicated()]
            nd = dfu[["f1","f2"]].to_numpy(dtype=float)

        if nd.size == 0:
            continue

        value = -nd[:, 0]   # maximize value -> plot -f1
        risk  =  nd[:, 1]   # minimize risk -> plot f2
        plt.scatter(risk, value,
                    s=14, alpha=0.9,
                    label=method,
                    color=colors.get(method, None),
                    marker=markers.get(method, "o"),
                    edgecolors="none")
        any_points = True

    # diagonal reference (visual cue only)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    diag_x = np.linspace(xmin, xmax, 200)
    diag_y = np.linspace(ymin, ymax, 200)
    plt.plot(diag_x, np.interp(diag_x, [xmin, xmax], [ymin, ymax]),
             color="lightgray", linewidth=2, alpha=0.35)

    title = f"Union of Non-Dominated Pareto Fronts for N={N}"
    if not any_points:
        title += " (No valid points)"
    plt.title(title)
    plt.xlabel("Risk ($f_2$)")
    plt.ylabel("Value ($-f_1$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hv_box(df_metrics, N, out_path):
    """Box plot for HV per method."""
    plt.figure(figsize=(8, 6))
    df_n = df_metrics[df_metrics['N'] == N]
    if df_n.empty:
        plt.title(f'Hypervolume (HV) Distribution for N={N} (No data)')
        plt.savefig(out_path)
        plt.close()
        return

    methods = df_n['method'].unique()
    data = [df_n[df_n['method'] == m]['HV'] for m in methods]

    plt.boxplot(data, labels=methods)
    plt.title(f'Hypervolume (HV) Distribution for N={N}')
    plt.ylabel('HV')
    plt.grid(True, axis='y')
    plt.savefig(out_path)
    plt.close()

def plot_runtime(meta_df, out_path):
    """Bar/box plot over elapsed_s by {N, method}."""
    if meta_df.empty:
        plt.figure(figsize=(10, 6))
        plt.title('Runtime Overview (No data)')
        plt.savefig(out_path)
        plt.close()
        return

    meta_df['N'] = meta_df['N'].astype(str)
    methods = meta_df['method'].unique()
    n_methods = len(methods)
    n_groups = len(meta_df['N'].unique())

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(n_groups)

    for i, method in enumerate(methods):
        method_data = meta_df[meta_df['method'] == method]
        means = method_data.groupby('N')['elapsed_s'].mean()
        stds = method_data.groupby('N')['elapsed_s'].std().fillna(0)

        # Align data with all N values
        all_ns = sorted(meta_df['N'].unique())
        aligned_means = [means.get(n, 0) for n in all_ns]
        aligned_stds = [stds.get(n, 0) for n in all_ns]

        ax.bar(index + i * bar_width, aligned_means, bar_width, yerr=aligned_stds, capsize=5, label=method)

    # Time cap line
    if 'time_cap_s' in meta_df.columns and not meta_df['time_cap_s'].isnull().all():
        time_cap = meta_df['time_cap_s'].max()
        ax.axhline(y=time_cap, color='r', linestyle='--', label=f'Time Cap ({time_cap}s)')

    ax.set_xlabel('Problem Size (N)')
    ax.set_ylabel('Elapsed Time (s)')
    ax.set_title('Runtime by Problem Size and Method')
    ax.set_xticks(index + bar_width / (2/n_methods) - bar_width/2)
    ax.set_xticklabels(sorted(meta_df['N'].unique()))
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_box_multi(df_metrics, N, out_path):
    """
    Generates a combined boxplot figure for HV, IGD, and ND_size for a given N.
    """
    df_n = df_metrics[df_metrics['N'] == N]
    if df_n.empty:
        print(f"No data for N={N} in plot_box_multi.")
        return

    metrics = ['HV', 'IGD', 'ND_size']
    titles = ['HV Distribution', 'IGD Distribution', '|ND| Distribution']
    methods = sorted(df_n['method'].unique())
    colors = {'NSGA2': '#1f77b4', 'RANDOM': '#ff7f0e'}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        data = [df_n[df_n['method'] == m][metric].dropna() for m in methods]
        
        bp = ax.boxplot(data, patch_artist=True, labels=methods)
        
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(colors.get(method, 'gray'))
            
        ax.set_title(title)
        ax.set_xlabel('Method')
        ax.set_ylabel(metric)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
