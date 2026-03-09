import os; os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Global Font Settings for Paper Quality
TITLE_FONT = {'fontsize': 18, 'fontweight': 'bold'}
LABEL_FONT = {'fontsize': 16, 'fontweight': 'bold'}
LEGEND_PROP = {'size': 14, 'weight': 'bold'}
TICK_FONT_SIZE = 14

def _apply_standard_style(ax, title, xlabel, ylabel, has_legend=True):
    ax.set_title(title, fontdict=TITLE_FONT)
    ax.set_xlabel(xlabel, fontdict=LABEL_FONT)
    ax.set_ylabel(ylabel, fontdict=LABEL_FONT)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    if has_legend:
        ax.legend(prop=LEGEND_PROP)
    ax.grid(True, alpha=0.3)

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

def plot_population_only(all_solutions_dir, N, out_path):
    """Plots all evaluated solutions across all seeds (sampled)."""
    import glob
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    color_map = {"NSGA2": "#1f77b4", "RANDOM": "#ff7f0e"}
    all_pattern = os.path.join(all_solutions_dir, f"all_2obj_*_N{N}_seed*.csv")
    all_files = glob.glob(all_pattern)
    
    method_data = {}
    for file_path in all_files:
        try:
            fname = os.path.basename(file_path)
            method = fname.split('_')[2].upper()
            df = pd.read_csv(file_path, usecols=["f1 (Return)", "f2 (Risk)"])
            df = df.rename(columns={"f1 (Return)": "f1", "f2 (Risk)": "f2"})
            if method not in method_data: method_data[method] = []
            method_data[method].append(df)
        except: continue

    any_plotted = False
    for method, dfs in method_data.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        if len(combined_df) > 20000: combined_df = combined_df.sample(20000, random_state=42)
        ax.scatter(combined_df['f2'], -combined_df['f1'], s=4, alpha=0.2, label=f"{method} (All)", color=color_map.get(method))
        any_plotted = True
    
    if any_plotted:
        _apply_standard_style(ax, f"All Evaluated Solutions (N={N})", "Expected Risk", "Expected Return")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pareto_only(df_fronts, N, out_path):
    """Plots the union of non-dominated fronts across all seeds."""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    color_map = {"NSGA2": "#1f77b4", "RANDOM": "#ff7f0e"}
    markers = {"NSGA2": "o", "RANDOM": "^"}
    
    df_n = df_fronts[df_fronts["N"] == N].copy()
    if df_n.empty:
        plt.close()
        return

    df_n["method"] = df_n["method"].astype(str).str.strip().str.upper()
    any_plotted = False
    for method, grp in df_n.groupby("method"):
        pts_all = grp[["f1","f2"]].dropna().to_numpy(dtype=float)
        nd = _union_nd_front(pts_all)
        if nd.size == 0: continue
        
        value = -nd[:, 0]
        risk  =  nd[:, 1]
        ax.scatter(risk, value, s=40, alpha=0.8, label=f"{method} (Pareto)", 
                   color=color_map.get(method), marker=markers.get(method, "o"))
        idx_sort = np.argsort(risk)
        ax.plot(risk[idx_sort], value[idx_sort], color=color_map.get(method), alpha=0.5, linestyle='--')
        any_plotted = True

    if any_plotted:
        _apply_standard_style(ax, f"Union of Pareto Fronts (N={N})", "Expected Risk", "Expected Return")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_optimization_process(all_solutions_dir, df_fronts, N, out_path):
    """Multi-panel aggregated plot (Top: All, Bottom: Pareto)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 16))
    
    color_map = {"NSGA2": "#1f77b4", "RANDOM": "#ff7f0e"}
    markers = {"NSGA2": "o", "RANDOM": "^"}
    
    # --- Top: All ---
    import glob
    all_pattern = os.path.join(all_solutions_dir, f"all_2obj_*_N{N}_seed*.csv")
    all_files = glob.glob(all_pattern)
    method_data = {}
    for file_path in all_files:
        try:
            fname = os.path.basename(file_path)
            method = fname.split('_')[2].upper()
            df = pd.read_csv(file_path, usecols=["f1 (Return)", "f2 (Risk)"])
            df = df.rename(columns={"f1 (Return)": "f1", "f2 (Risk)": "f2"})
            if method not in method_data: method_data[method] = []
            method_data[method].append(df)
        except: continue

    for method, dfs in method_data.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        if len(combined_df) > 20000: combined_df = combined_df.sample(20000, random_state=42)
        axes[0].scatter(combined_df['f2'], -combined_df['f1'], s=3, alpha=0.15, label=f"{method} (All)", color=color_map.get(method))
    
    _apply_standard_style(axes[0], f"All Evaluated Solutions (N={N})", "Expected Risk", "Expected Return")

    # --- Bottom: Pareto ---
    df_n = df_fronts[df_fronts["N"] == N].copy()
    if not df_n.empty:
        df_n["method"] = df_n["method"].astype(str).str.strip().str.upper()
        for method, grp in df_n.groupby("method"):
            pts_all = grp[["f1","f2"]].dropna().to_numpy(dtype=float)
            nd = _union_nd_front(pts_all)
            if nd.size == 0: continue
            value, risk = -nd[:, 0], nd[:, 1]
            axes[1].scatter(risk, value, s=30, alpha=0.8, label=f"{method} (Pareto)", 
                            color=color_map.get(method), marker=markers.get(method, "o"))
            idx_sort = np.argsort(risk)
            axes[1].plot(risk[idx_sort], value[idx_sort], color=color_map.get(method), alpha=0.4, linestyle='--')
    
    _apply_standard_style(axes[1], f"Union of Pareto Fronts (N={N})", "Expected Risk", "Expected Return")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_metric_box(df_metrics, N, metric, title, ylabel, out_path):
    """Generic boxplot for a single metric."""
    plt.figure(figsize=(8, 8))
    df_n = df_metrics[df_metrics['N'] == N]
    if df_n.empty:
        plt.close()
        return

    methods = sorted(df_n['method'].unique())
    colors = {'NSGA2': '#1f77b4', 'RANDOM': '#ff7f0e'}
    data = [df_n[df_n['method'] == m][metric].dropna() for m in methods]

    ax = plt.gca()
    bp = ax.boxplot(data, patch_artist=True, labels=methods)
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors.get(method, 'gray'))
    
    # Add N to title
    full_title = f"{title} (N={N})"
    _apply_standard_style(ax, full_title, "Method", ylabel, has_legend=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_box_multi(df_metrics, N, out_path):
    """Generates a combined vertical boxplot figure for HV, IGD, and ND_size."""
    df_n = df_metrics[df_metrics['N'] == N]
    if df_n.empty: return

    metrics = ['HV', 'IGD', 'ND_size']
    titles = [f'HV Distribution (N={N})', f'IGD Distribution (N={N})', f'|ND| Distribution (N={N})']
    ylabels = ['Hypervolume', 'IGD+', 'Number of ND points']
    methods = sorted(df_n['method'].unique())
    colors = {'NSGA2': '#1f77b4', 'RANDOM': '#ff7f0e'}

    fig, axes = plt.subplots(3, 1, figsize=(8, 18))
    for i, (m_key, title, yl) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        data = [df_n[df_n['method'] == m][m_key].dropna() for m in methods]
        bp = ax.boxplot(data, patch_artist=True, labels=methods)
        for patch, method in zip(bp['boxes'], methods):
            patch.set_facecolor(colors.get(method, 'gray'))
        _apply_standard_style(ax, title, "Method", yl, has_legend=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_runtime(meta_df, out_path):
    """Bar plot over elapsed_s by {N, method}, sorted by numeric N."""
    if meta_df.empty: return
    meta_df['N_numeric'] = pd.to_numeric(meta_df['N'], errors='coerce')
    all_ns_numeric = sorted(meta_df['N_numeric'].unique())
    all_ns_labels = [str(int(n)) for n in all_ns_numeric]
    methods = sorted(meta_df['method'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(all_ns_numeric))

    for i, method in enumerate(methods):
        method_data = meta_df[meta_df['method'] == method]
        means = method_data.groupby('N_numeric')['elapsed_s'].mean()
        stds = method_data.groupby('N_numeric')['elapsed_s'].std().fillna(0)
        ax.bar(index + i * bar_width, [means.get(n, 0) for n in all_ns_numeric], 
               bar_width, yerr=[stds.get(n, 0) for n in all_ns_numeric], 
               capsize=5, label=method)

    _apply_standard_style(ax, "Runtime by Problem Size and Method", "Problem Size (N)", "Elapsed Time (s)")
    ax.set_xticks(index + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(all_ns_labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
