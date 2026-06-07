"""
Script 4: open_close Ablation + DT Depth Sensitivity
Reads directly from: results/cross_dataset.csv and results/cross_dataset_aggregated.csv
"""
import numpy as np
import pandas as pd
from tabulate import tabulate

RESULTS_DIR = "results"

if __name__ == "__main__":
    agg = pd.read_csv(f"{RESULTS_DIR}/cross_dataset_aggregated.csv")
    cd  = pd.read_csv(f"{RESULTS_DIR}/cross_dataset.csv")

    # ── Experiment A: open_close ablation ────────────────────────────────────
    # Proxy: compare ctx (includes open_close) vs full (adds demo features)
    # True ablation needs a ctx_no_oc run — check if that exists in results
    print("=" * 65)
    print("Experiment A: open_close Source-Indicator Check")
    print("=" * 65)
    print()

    # Check if a ctx_no_oc feature set exists in results
    feat_sets = agg.features.unique()
    print(f"  Feature sets in results: {list(feat_sets)}")

    if "ctx_no_oc" in feat_sets:
        def min_r(df, feat):
            return df[df.features==feat].groupby(["strategy","model","threshold_mode"])["recall_very_high_mean"].min()
        ctx_min    = min_r(agg, "ctx")
        no_oc_min  = min_r(agg, "ctx_no_oc")
        best_key   = ("quartile","decision_tree","default")
        ctx_val    = ctx_min.get(best_key, np.nan)
        no_oc_val  = no_oc_min.get(best_key, np.nan)
        delta      = ctx_val - no_oc_val
        print(f"  Best config (quartile/DT/default):")
        print(f"    ctx (with open_close) min recall : {ctx_val:.3f}")
        print(f"    ctx_no_oc             min recall : {no_oc_val:.3f}")
        print(f"    Δ                                : {delta:+.3f}")
        if abs(delta) < 0.05:
            print("  Verdict: ROBUST — open_close does not act as source identifier.")
        elif delta > 0:
            print(f"  Verdict: CONFOUND LIKELY — removing open_close drops recall by {delta:.3f}.")
        else:
            print(f"  Verdict: NO CONFOUND — removing open_close improves recall by {abs(delta):.3f}.")
    else:
        print("  ctx_no_oc feature set not found in results.")
        print("  To run this ablation: re-run your pipeline with features=['incident_area','gender']")
        print("  (dropping open_close) and save to results/cross_dataset_aggregated.csv.")
        print()
        # Proxy: check how much open_close contributes via feature importance
        try:
            fi = pd.read_csv(f"{RESULTS_DIR}/feature_importance.csv")
            oc_feats = fi[fi.feature.str.startswith("open_close")]
            if len(oc_feats) > 0:
                print("  open_close feature importances (full feature set, std/DT/default):")
                print(oc_feats[["feature","importance_mean","importance_std"]].to_string(index=False))
                total_oc = oc_feats.importance_mean.sum()
                print(f"\n  Total open_close importance: {total_oc:.4f}")
                if total_oc < 0.005:
                    print("  LOW importance — open_close unlikely to act as source indicator.")
                else:
                    print("  NON-TRIVIAL importance — ablation run recommended.")
        except FileNotFoundError:
            pass

    # ── Experiment B: DT depth sensitivity ───────────────────────────────────
    print()
    print("=" * 65)
    print("Experiment B: DT Depth Sensitivity")
    print("  (from results — checks if multiple depths were tested)")
    print("=" * 65)
    print()

    # Check if depth is encoded in results (it's not by default — flag this)
    print("  Depth column not present in cross_dataset_aggregated.csv.")
    print("  To justify depth=5, re-run your LODO pipeline with:")
    depths = [3, 4, 5, 6, 7, None]
    print(f"  depths = {depths}")
    print()
    print("  Suggested code addition to your pipeline:")
    print("""
    for depth in [3, 4, 5, 6, 7, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
        # run LODO, record min VeryHigh recall
        results[depth] = min_recall
    """)

    # What we CAN report from existing results: best config uses depth=5
    best_ctx = agg[(agg.features=="ctx")&(agg.model=="decision_tree")&(agg.threshold_mode=="default")]
    best_ctx_min = best_ctx.groupby(["strategy","model","threshold_mode"])["recall_very_high_mean"].min()
    print("  Best existing DT/default/ctx configs by min recall across sources:")
    per_strat = []
    for strat in ["quartile","std","rule"]:
        sub = agg[(agg.features=="ctx")&(agg.model=="decision_tree")&
                  (agg.threshold_mode=="default")&(agg.strategy==strat)]
        if len(sub) > 0:
            min_r = sub.groupby("test_source")["recall_very_high_mean"].mean()
            per_strat.append({"strategy":strat, "min_mean_recall":round(min_r.min(),3)})
    print(tabulate(per_strat, headers="keys", tablefmt="github"))
    print()
    print("  Paper note: 'Depth-5 was adopted following prior work on the Kaggle")
    print("  source; a depth sensitivity analysis (depths 3-7) is recommended")
    print("  to confirm this choice on the pooled multi-source training set.'")