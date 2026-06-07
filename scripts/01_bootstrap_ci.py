"""
Script 1: Bootstrap CIs on LODO VeryHigh Recall
Reads directly from: results/cross_dataset.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

RESULTS_DIR = "results"
TOP_CONFIGS = [
    ("quartile", "decision_tree",  "default",   "ctx"),
    ("std",      "multinomial_lr", "youden_val","ctx"),
    ("std",      "decision_tree",  "default",   "ctx"),
    ("std",      "decision_tree",  "default",   "full"),
    ("std",      "naive_bayes",    "youden",    "full"),
]
SOURCE_MAP = {"kaggle":"Kaggle","mother_jones":"Mother Jones","stanford_msa":"Stanford MSA"}

def bootstrap_ci(values, n_resamples=10_000, random_state=42):
    values = np.asarray(values, dtype=float)
    res = bootstrap((values,), np.mean, n_resamples=n_resamples,
                    confidence_level=0.95, method="percentile",
                    random_state=random_state)
    return float(np.mean(values)), float(res.confidence_interval.low), float(res.confidence_interval.high)

def min_recall_per_seed(per_source):
    sources = list(per_source.keys())
    n = len(per_source[sources[0]])
    return np.array([min(per_source[s][i] for s in sources) for i in range(n)])

if __name__ == "__main__":
    cd = pd.read_csv(f"{RESULTS_DIR}/cross_dataset.csv")

    print("=" * 65)
    print("Bootstrap 95% CIs — Minimum LODO VeryHigh Recall (5 seeds)")
    print("=" * 65)
    print()
    for strat, model, thr, feat in TOP_CONFIGS:
        sub = cd[(cd.strategy==strat)&(cd.model==model)&
                 (cd.threshold_mode==thr)&(cd.features==feat)]
        label = f"{strat} / {model} / {thr} / {feat}"
        per_source = {}
        for src, src_label in SOURCE_MAP.items():
            per_source[src_label] = sub[sub.test_source==src].sort_values("seed")["recall_very_high"].tolist()
        mins = min_recall_per_seed(per_source)
        mean, lo, hi = bootstrap_ci(mins)
        print(f"  {label}")
        print(f"    Per-seed minimums : {np.round(mins,4).tolist()}")
        print(f"    Mean min recall   : {mean:.3f}  (SD={np.std(mins):.3f})")
        print(f"    95% bootstrap CI  : [{lo:.3f}, {hi:.3f}]")
        print()

    print("=" * 65)
    print("Per-source breakdown (mean ± SD, 95% CI)")
    print("=" * 65)
    print()
    for strat, model, thr, feat in TOP_CONFIGS:
        sub = cd[(cd.strategy==strat)&(cd.model==model)&
                 (cd.threshold_mode==thr)&(cd.features==feat)]
        label = f"{strat} / {model} / {thr} / {feat}"
        print(f"  {label}")
        for src, src_label in SOURCE_MAP.items():
            vals = np.array(sub[sub.test_source==src].sort_values("seed")["recall_very_high"].tolist())
            mean, lo, hi = bootstrap_ci(vals)
            print(f"    {src_label:<15}: {mean:.3f} ± {np.std(vals):.3f}  CI [{lo:.3f}, {hi:.3f}]")
        print()