"""
Script 6: RF Baseline + DT Depth Sweep Pipeline Extension
==========================================================
Runs your existing LODO protocol with:
  - Random Forest added as a 4th classifier
  - DT depth sweep [3, 4, 5, 6, 7, None]

Reads raw data from:  results/cross_dataset.csv  (for source assignments)
Writes new results to: results/cross_dataset_aggregated_extended.csv

Requirements: scikit-learn, imbalanced-learn, pandas, numpy
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

RESULTS_DIR  = "results"
SEEDS        = [42, 43, 44, 45, 46]
SOURCES      = ["kaggle", "mother_jones", "stanford_msa"]
SOURCE_COL   = "source"
TOTAL_COL    = "total_victims"
TARGET_COL   = "risk_label"

# ── Load raw incident data ────────────────────────────────────────────────────
# The cross_dataset.csv has per-seed per-source results but not raw incidents.
# Point these paths to your harmonized source files.
# If you have a single pooled file with a 'source' column, set POOLED_CSV instead.

POOLED_CSV   = "data/processed/harmonized.csv"   # ← set this path
CTX_FEATURES = ["incident_area", "open_close", "gender"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def quartile_labels(series, thresholds):
    q1, q2, q3 = thresholds
    return pd.cut(series, bins=[-np.inf, q1, q2, q3, np.inf],
                  labels=["Low","Medium","High","VeryHigh"]).astype(str)

def encode(X_train, X_test, cols):
    combined = pd.concat([X_train[cols], X_test[cols]])
    dummies  = pd.get_dummies(combined, columns=cols, drop_first=False)
    n = len(X_train)
    return dummies.iloc[:n].values.astype(float), dummies.iloc[n:].values.astype(float)

def vh_recall(y_true, y_pred):
    mask = y_true == "VeryHigh"
    return (y_pred[mask] == "VeryHigh").sum() / mask.sum() if mask.sum() > 0 else np.nan

def run_lodo(df, clf_factory, feature_cols, seeds=SEEDS):
    """Returns DataFrame of per-seed per-source VeryHigh recall."""
    rows = []
    for seed in seeds:
        for held_out in SOURCES:
            train_df = df[df[SOURCE_COL] != held_out].copy()
            test_df  = df[df[SOURCE_COL] == held_out].copy()

            thresholds = (train_df[TOTAL_COL].quantile(0.25),
                          train_df[TOTAL_COL].quantile(0.50),
                          train_df[TOTAL_COL].quantile(0.75))

            train_df[TARGET_COL] = quartile_labels(train_df[TOTAL_COL], thresholds)
            test_df[TARGET_COL]  = quartile_labels(test_df[TOTAL_COL],  thresholds)
            train_df = train_df[train_df[TARGET_COL] != "nan"]
            test_df  = test_df[test_df[TARGET_COL]  != "nan"]

            if len(test_df) == 0 or "VeryHigh" not in test_df[TARGET_COL].values:
                rows.append({"seed":seed,"test_source":held_out,"recall_very_high":np.nan})
                continue

            X_tr, X_te = encode(train_df, test_df, feature_cols)
            y_tr = train_df[TARGET_COL].values
            y_te = test_df[TARGET_COL].values

            ros = RandomOverSampler(sampling_strategy="auto", random_state=seed)
            try:
                X_tr_os, y_tr_os = ros.fit_resample(X_tr, y_tr)
            except ValueError:
                X_tr_os, y_tr_os = X_tr, y_tr

            clf = clf_factory(seed)
            clf.fit(X_tr_os, y_tr_os)
            y_pred = clf.predict(X_te)
            rows.append({"seed":seed,"test_source":held_out,
                         "recall_very_high":vh_recall(y_te, y_pred)})
    return pd.DataFrame(rows)

def aggregate(df, label_cols):
    """Aggregate per-seed results to mean/std per source."""
    agg = df.groupby("test_source")["recall_very_high"].agg(["mean","std"]).reset_index()
    agg.columns = ["test_source","recall_very_high_mean","recall_very_high_std"]
    for k, v in label_cols.items():
        agg[k] = v
    return agg

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    if not os.path.exists(POOLED_CSV):
        print(f"ERROR: {POOLED_CSV} not found.")
        print("Set POOLED_CSV to your harmonized pooled dataset path.")
        print("The file needs columns: source, total_victims, incident_area,")
        print("                        open_close, gender")
        exit(1)

    df = pd.read_csv(POOLED_CSV)
    print(f"Loaded {len(df)} rows from {POOLED_CSV}")
    print(f"Sources: {df[SOURCE_COL].value_counts().to_dict()}\n")

    all_results = []

    # ── RF as 4th classifier (quartile / default / ctx) ──────────────────────
    print("Running RF baseline (quartile/default/ctx)...")
    rf_factory = lambda s: RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=s)
    rf_raw = run_lodo(df, rf_factory, CTX_FEATURES)
    rf_agg = aggregate(rf_raw, {"strategy":"quartile","model":"random_forest",
                                 "threshold_mode":"default","features":"ctx"})
    all_results.append(rf_agg)
    min_r = rf_agg.recall_very_high_mean.min()
    print(f"  RF min cross-source recall: {min_r:.3f}")
    print(rf_agg[["test_source","recall_very_high_mean","recall_very_high_std"]].to_string(index=False))
    print()

    # ── DT depth sweep (quartile / default / ctx) ────────────────────────────
    print("Running DT depth sweep [3, 4, 5, 6, 7, None]...")
    depth_rows = []
    for depth in [3, 4, 5, 6, 7, None]:
        label = str(depth) if depth is not None else "unconstrained"
        dt_factory = lambda s, d=depth: DecisionTreeClassifier(max_depth=d, random_state=s)
        raw = run_lodo(df, dt_factory, CTX_FEATURES)
        agg_row = aggregate(raw, {"strategy":"quartile",
                                   "model":f"decision_tree_d{label}",
                                   "threshold_mode":"default","features":"ctx"})
        all_results.append(agg_row)
        min_r = agg_row.recall_very_high_mean.min()
        depth_rows.append({"depth":label, "min_recall":round(min_r,3)})
        print(f"  depth={label:>12}: min recall = {min_r:.3f}")

    print()
    best = max(depth_rows, key=lambda r: r["min_recall"])
    print(f"Best depth by min recall: {best['depth']} (min = {best['min_recall']:.3f})")
    if best["depth"] == "5":
        print("✓ depth=5 is empirically justified.")
    else:
        print(f"✗ depth=5 is NOT optimal. Best is depth={best['depth']}.")
    print()

    # ── Save extended results ─────────────────────────────────────────────────
    out_path = f"{RESULTS_DIR}/cross_dataset_aggregated_extended.csv"
    pd.concat(all_results, ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved extended results to {out_path}")
    print("Run scripts 04 and 05 again to pick up RF and depth results.")