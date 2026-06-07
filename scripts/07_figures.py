"""
Script 7: Paper Figures (modified for single-column IEEE layout)
=================================================================
Generates 4 figures recommended for the paper.

Changes from original:
- Figure 1: 3 panels merged into a single grouped horizontal bar chart
            (5 configs × 3 sources), sized for IEEE single column.
- Figure 2: heatmap transposed (rows = model/threshold, cols = strategy),
            sized for IEEE single column.
- Figures 3 and 4: unchanged.

Reads from: results/cross_dataset.csv
            results/cross_dataset_aggregated.csv
            results/feature_importance_triangulation.csv

Output: figures/fig1_bootstrap_ci.pdf
        figures/fig2_paired_heatmap.pdf
        figures/fig3_feature_importance.pdf
        figures/fig4_youden_breakdown.pdf

Requirements: matplotlib, numpy, pandas, scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import bootstrap

os.makedirs("figures", exist_ok=True)
RESULTS_DIR = "results"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
cd  = pd.read_csv(f"{RESULTS_DIR}/cross_dataset.csv")
agg = pd.read_csv(f"{RESULTS_DIR}/cross_dataset_aggregated.csv")
fi  = pd.read_csv(f"{RESULTS_DIR}/feature_importance_triangulation.csv")

SOURCE_MAP = {"kaggle":"Kaggle","mother_jones":"Mother Jones","stanford_msa":"Stanford MSA"}

def bci(values, n_resamples=5000, random_state=42):
    values = np.asarray(values, dtype=float)
    res = bootstrap((values,), np.mean, n_resamples=n_resamples,
                    confidence_level=0.95, method="percentile",
                    random_state=random_state)
    return np.mean(values), res.confidence_interval.low, res.confidence_interval.high

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 (MODIFIED): single-panel grouped horizontal bar chart
# Best figure type: grouped horizontal bars, 5 configs × 3 sources
# Why: same information as the original 3-panel version but ~3x narrower,
#      letting it fit in a single IEEE column and saving vertical space.
#      Direct per-source comparison is preserved via color grouping.
# ══════════════════════════════════════════════════════════════════════════════

TOP_CONFIGS = [
    ("quartile","decision_tree","default","ctx",   "quartile / DT / default (ctx)"),
    ("std","multinomial_lr","youden_val","ctx",     "std / MLR / Youden-val (ctx)"),
    ("std","decision_tree","default","ctx",         "std / DT / default (ctx)"),
    ("std","decision_tree","default","full",        "std / DT / default (full)"),
    ("std","naive_bayes","youden","full",           "std / GNB / Youden (full)"),
]
COLORS = {"Kaggle":"#2196F3","Mother Jones":"#E91E63","Stanford MSA":"#4CAF50"}

n_configs = len(TOP_CONFIGS)
source_keys   = list(SOURCE_MAP.keys())
source_labels = list(SOURCE_MAP.values())

means_arr = np.zeros((n_configs, len(source_keys)))
los_arr   = np.zeros((n_configs, len(source_keys)))
his_arr   = np.zeros((n_configs, len(source_keys)))

for c_idx, (strat, model, thr, feat, _) in enumerate(TOP_CONFIGS):
    for s_idx, src_key in enumerate(source_keys):
        sub = cd[(cd.strategy==strat)&(cd.model==model)&
                 (cd.threshold_mode==thr)&(cd.features==feat)&
                 (cd.test_source==src_key)]
        vals = sub.sort_values("seed")["recall_very_high"].values
        m, lo, hi = bci(vals)
        means_arr[c_idx, s_idx] = m
        los_arr[c_idx, s_idx]   = m - lo
        his_arr[c_idx, s_idx]   = hi - m

config_labels = [label for *_, label in TOP_CONFIGS]
bar_h = 0.25
y = np.arange(n_configs)

# IEEE single-column width ~3.5 in
fig, ax = plt.subplots(figsize=(3.5, 3.4))

for i, src_label in enumerate(source_labels):
    offset = (i - 1) * bar_h  # center the three bars around each y tick
    ax.barh(y + offset, means_arr[:, i], height=bar_h,
            xerr=[los_arr[:, i], his_arr[:, i]],
            color=COLORS[src_label], alpha=0.85, label=src_label,
            error_kw={"ecolor":"#333","capsize":1.8,"linewidth":0.7})

ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax.set_yticks(y)
ax.set_yticklabels(config_labels, fontsize=7.5)
ax.set_xlabel("VeryHigh Recall", fontsize=8.5)
ax.set_xlim(0, 1.15)
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9, handlelength=1.2)

plt.tight_layout()
plt.savefig("figures/fig1_bootstrap_ci.pdf", bbox_inches="tight")
plt.savefig("figures/fig1_bootstrap_ci.png", bbox_inches="tight")
plt.close()
print("Saved fig1_bootstrap_ci (single-panel merged)")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 (MODIFIED): transposed heatmap (rows = model/threshold, cols = strategy)
# Best figure type: tall heatmap with diverging colormap
# Why: same information as the original 3×9 wide layout but oriented vertically
#      so it fits in a single IEEE column. All 27 paired comparisons remain visible.
# ══════════════════════════════════════════════════════════════════════════════

def min_recall(df, feat):
    return (df[df.features==feat]
            .groupby(["strategy","model","threshold_mode"])
            ["recall_very_high_mean"].min())

ctx_min  = min_recall(agg, "ctx")
full_min = min_recall(agg, "full")

strategies   = ["quartile","std","rule"]
models       = ["decision_tree","multinomial_lr","naive_bayes"]
thresholds   = ["default","youden","youden_val"]
model_labels = {"decision_tree":"DT","multinomial_lr":"MLR","naive_bayes":"GNB"}

# Build transposed delta array: rows = model/threshold (9), cols = strategy (3)
n_rows = len(models) * len(thresholds)
n_cols = len(strategies)
delta_data = np.full((n_rows, n_cols), np.nan)
row_labels = []
for m_idx, model in enumerate(models):
    for t_idx, thr in enumerate(thresholds):
        row = m_idx * len(thresholds) + t_idx
        row_labels.append(f"{model_labels[model]} {thr}")
        for s_idx, strat in enumerate(strategies):
            key = (strat, model, thr)
            ctx_val  = ctx_min.get(key, np.nan)
            full_val = full_min.get(key, np.nan)
            if not (np.isnan(ctx_val) or np.isnan(full_val)):
                delta_data[row, s_idx] = ctx_val - full_val

col_labels = [s.capitalize() for s in strategies]

# IEEE single-column width ~3.5 in; tall aspect to fit 9 rows
fig, ax = plt.subplots(figsize=(3.5, 4.6))
im = ax.imshow(delta_data, cmap="RdBu", vmin=-0.7, vmax=0.7, aspect="auto")

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=8)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=7.5)
ax.set_xlabel("Strategy", fontsize=8.5)
ax.set_ylabel("Model / Threshold", fontsize=8.5)

# Annotate cells
for i in range(delta_data.shape[0]):
    for j in range(delta_data.shape[1]):
        val = delta_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(val) > 0.35 else "black")

cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
cbar.set_label(r"$\Delta$ recall", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("figures/fig2_paired_heatmap.pdf", bbox_inches="tight")
plt.savefig("figures/fig2_paired_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved fig2_paired_heatmap (transposed)")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Feature importance triangulation — grouped bar chart (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

METHODS = ["permutation_primary","permutation_rf","gini_rf"]
METHOD_LABELS = {"permutation_primary":"Permutation\n(primary DT)",
                 "permutation_rf":"Permutation\n(Random Forest)",
                 "gini_rf":"Gini decrease\n(Random Forest)"}
METHOD_COLORS = {"permutation_primary":"#1976D2",
                 "permutation_rf":"#388E3C",
                 "gini_rf":"#F57C00"}

# Top 8 features by max importance across any method
top_feats = (fi.groupby("feature")["importance_mean"]
               .max().nlargest(8).index.tolist())

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
fig.suptitle("Feature Importance Triangulation (std / DT / default, full features)",
             fontsize=10, fontweight="bold")

for ax, method in zip(axes, METHODS):
    sub = fi[fi.method==method].set_index("feature")
    vals = [max(sub.loc[f,"importance_mean"], 0) if f in sub.index else 0
            for f in top_feats]
    errs = [sub.loc[f,"importance_std"] if f in sub.index else 0
            for f in top_feats]
    y = np.arange(len(top_feats))
    ax.barh(y, vals, xerr=errs, height=0.6,
            color=METHOD_COLORS[method], alpha=0.8,
            error_kw={"ecolor":"#555","capsize":2,"linewidth":0.8})
    ax.set_title(METHOD_LABELS[method], fontsize=8.5)
    ax.set_xlabel("Importance", fontsize=8)
    ax.axvline(0, color="gray", linewidth=0.5)
    if ax == axes[0]:
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f.replace("incident_area_","ia_")
              .replace("mental_health_","mh_")
              .replace("open_close_","oc_")
             for f in top_feats], fontsize=7.5)

plt.tight_layout()
plt.savefig("figures/fig3_feature_importance.pdf", bbox_inches="tight")
plt.savefig("figures/fig3_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved fig3_feature_importance")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Youden vs Default — per-source recall breakdown (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

sources_ordered = ["kaggle","mother_jones","stanford_msa"]
src_labels      = ["Kaggle","Mother Jones","Stanford MSA"]

def_vals, yod_vals = [], []
for src in sources_ordered:
    d = agg[(agg.strategy=="std")&(agg.model=="multinomial_lr")&
            (agg.threshold_mode=="default")&(agg.features=="ctx")&
            (agg.test_source==src)]
    y = agg[(agg.strategy=="std")&(agg.model=="multinomial_lr")&
            (agg.threshold_mode=="youden_val")&(agg.features=="ctx")&
            (agg.test_source==src)]
    def_vals.append(d["recall_very_high_mean"].values[0] if len(d)>0 else 0)
    yod_vals.append(y["recall_very_high_mean"].values[0] if len(y)>0 else 0)

x     = np.arange(len(src_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bars1 = ax.bar(x - width/2, def_vals, width, label="Default threshold",
               color="#90CAF9", edgecolor="#1565C0", linewidth=0.8)
bars2 = ax.bar(x + width/2, yod_vals, width, label="Youden-val threshold",
               color="#1976D2", edgecolor="#0D47A1", linewidth=0.8)

# Annotate delta
for xi, (d, y) in enumerate(zip(def_vals, yod_vals)):
    delta = y - d
    sign  = "+" if delta >= 0 else ""
    color = "#2E7D32" if delta >= 0 else "#C62828"
    ax.annotate(f"{sign}{delta:.3f}",
                xy=(xi, max(d, y) + 0.02),
                ha="center", fontsize=8, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(src_labels, fontsize=9)
ax.set_ylabel("VeryHigh Recall (mean, 5 seeds)")
ax.set_ylim(0, 1.1)
ax.set_title("Youden-val vs Default Thresholding\n(std / MLR / ctx features)",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

plt.tight_layout()
plt.savefig("figures/fig4_youden_breakdown.pdf", bbox_inches="tight")
plt.savefig("figures/fig4_youden_breakdown.png", bbox_inches="tight")
plt.close()
print("Saved fig4_youden_breakdown")

print("\nAll figures saved to figures/")
print("Include in LaTeX with:")
print(r"  \begin{figure}[!t]")
print(r"  \centering")
print(r"  \includegraphics[width=\columnwidth]{figures/fig1_bootstrap_ci.pdf}")
print(r"  \caption{...}")
print(r"  \label{fig:bsci}")
print(r"  \end{figure}")