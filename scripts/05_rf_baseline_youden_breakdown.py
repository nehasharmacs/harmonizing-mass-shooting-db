"""
Script 5: RF Baseline Check + Per-Source Youden vs Default Breakdown
Reads directly from: results/cross_dataset_aggregated.csv
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

RESULTS_DIR = "results"

if __name__ == "__main__":
    agg = pd.read_csv(f"{RESULTS_DIR}/cross_dataset_aggregated.csv")

    SOURCES   = ["kaggle","mother_jones","stanford_msa"]
    SRC_SHORT = {"kaggle":"Kag","mother_jones":"MJ","stanford_msa":"Stan"}

    def min_recall_row(agg, strat, model, thr, feat, label):
        sub = agg[(agg.strategy==strat)&(agg.model==model)&
                  (agg.threshold_mode==thr)&(agg.features==feat)]
        if len(sub) == 0:
            return None
        per_src = {}
        for src in SOURCES:
            row = sub[sub.test_source==src]
            per_src[SRC_SHORT[src]] = round(row["recall_very_high_mean"].values[0],3) if len(row)>0 else np.nan
        min_r = min(per_src.values())
        return {"config": label, "min_recall": round(min_r,3), **per_src}

    # ── Experiment C: classifier comparison ──────────────────────────────────
    print("=" * 65)
    print("Experiment C: Classifier Comparison (quartile/default/ctx)")
    print("=" * 65)
    print()

    models = [("decision_tree","DT (depth=5)"),
              ("multinomial_lr","MLR (L2)"),
              ("naive_bayes","GNB")]

    rows_c = []
    for model, label in models:
        row = min_recall_row(agg,"quartile",model,"default","ctx",label)
        if row:
            rows_c.append(row)
            print(f"  {label:20s}: min={row['min_recall']:.3f}  "
                  + "  ".join(f"{k}={v:.3f}" for k,v in row.items() if k in SRC_SHORT.values()))

    # Check if RF is in results
    rf_row = min_recall_row(agg,"quartile","random_forest","default","ctx","RF (200 trees)")
    if rf_row:
        rows_c.append(rf_row)
        print(f"  {'RF (200 trees)':20s}: min={rf_row['min_recall']:.3f}  "
              + "  ".join(f"{k}={v:.3f}" for k,v in rf_row.items() if k in SRC_SHORT.values()))
    else:
        print()
        print("  Random Forest not found in results.")
        print("  To add RF as a 4th classifier, re-run your pipeline with:")
        print("    RandomForestClassifier(n_estimators=200, max_depth=10,")
        print("                           class_weight='balanced', random_state=seed)")

    print()
    print(tabulate(rows_c, headers="keys", tablefmt="github", floatfmt=".3f"))

    # ── Experiment D: Youden vs default per-source breakdown ─────────────────
    print()
    print("=" * 65)
    print("Experiment D: Youden-val vs Default — Per-Source Breakdown")
    print("  Config: std / MLR / ctx")
    print("=" * 65)
    print()

    rows_d = []
    print(f"  {'Source':<20} {'Default':>10} {'Youden-val':>12} {'Δ':>8}")
    print("  " + "-" * 52)
    for src in SOURCES:
        def_row = agg[(agg.strategy=="std")&(agg.model=="multinomial_lr")&
                      (agg.threshold_mode=="default")&(agg.features=="ctx")&
                      (agg.test_source==src)]
        yod_row = agg[(agg.strategy=="std")&(agg.model=="multinomial_lr")&
                      (agg.threshold_mode=="youden_val")&(agg.features=="ctx")&
                      (agg.test_source==src)]
        def_val = def_row["recall_very_high_mean"].values[0] if len(def_row)>0 else np.nan
        yod_val = yod_row["recall_very_high_mean"].values[0] if len(yod_row)>0 else np.nan
        delta   = yod_val - def_val
        helped  = "✓" if delta > 0.02 else ("~" if abs(delta) <= 0.02 else "✗")
        rows_d.append({"source":src,"default":round(def_val,3),
                       "youden_val":round(yod_val,3),"delta":round(delta,3),"helped?":helped})
        print(f"  {src:<20} {def_val:>10.3f} {yod_val:>12.3f} {delta:>+8.3f}")

    print()
    print(tabulate(rows_d, headers="keys", tablefmt="github"))
    print()

    mj_delta = next((r["delta"] for r in rows_d if "jones" in r["source"]), 0)
    other_d  = [r["delta"] for r in rows_d if "jones" not in r["source"]]
    if mj_delta > 0 and mj_delta >= max(other_d):
        print("  Finding: Youden-val disproportionately benefits Mother Jones")
        print("  (largest Δ), confirming it primarily rescues the distributional")
        print("  shift described in Section II-B. Add to Section V-A.")
    elif all(d > 0 for d in [mj_delta]+other_d):
        print("  Finding: Youden-val improves recall across ALL sources.")
        print("  Broaden the claim in Section V-A accordingly.")
    else:
        print("  Finding: Mixed results — report per-source deltas accurately.")

    # ── Feature importance summary ────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Feature Importance Summary (from feature_importance_triangulation.csv)")
    print("=" * 65)
    try:
        fi = pd.read_csv(f"{RESULTS_DIR}/feature_importance_triangulation.csv")
        for method in fi.method.unique():
            sub = fi[fi.method==method].sort_values("importance_mean",ascending=False).head(5)
            print(f"\n  Top 5 — {method}:")
            print(sub[["feature","importance_mean","importance_std"]].to_string(index=False))
    except FileNotFoundError:
        print("  feature_importance_triangulation.csv not found in results/")