"""
run_experiments.py

Main experiment driver. Produces:
  results/within_dataset.csv   — pooled 5-fold CV across all sources
                                 (3 strategies x 3 models x 2 threshold modes)
  results/cross_dataset.csv    — leave-one-dataset-out generalization
  results/feature_importance.csv — permutation importance for best config
  results/dataset_summary.csv  — per-source descriptive stats (from preprocess)

Usage:
    python -m src.run_experiments --output-dir results/
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .classification_strategies import LABELS
from .evaluation import (
    ConfigResult,
    FoldResult,
    feature_importance,
    run_cross_dataset,
    run_cv,
)
from .models import MODELS
from .temporal_holdout import aggregate_temporal, temporal_holdout_sweep

# ---------------------------------------------------------------------------
# Feature sets (matches the feature-ablation the original project tried)
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    # Minimal set — contextual features only
    "ctx":     ["incident_area", "open_close", "gender"],
    # Full set — contextual + demographic + clinical indicator
    "full":    ["incident_area", "open_close", "gender",
                "age", "mental_health", "race", "multiple_shooters"],
}

RISK_COLUMNS = {
    "rule":     "risk_rule",
    "std":      "risk_std",
    "quartile": "risk_quartile",
}


def _encode(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """One-hot encode categorical columns; keep numeric as-is."""
    num = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feature_cols if c not in num]
    X_cat = pd.get_dummies(df[cat], drop_first=True) if cat else pd.DataFrame(index=df.index)
    X_num = df[num].astype(float) if num else pd.DataFrame(index=df.index)
    return pd.concat([X_cat, X_num], axis=1).astype(float)


def within_dataset_sweep(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the 3 x 3 x 2 x |feature_sets| sweep with stratified CV."""
    rows = []
    total = (len(RISK_COLUMNS) * len(MODELS)
             * 2 * len(FEATURE_SETS))
    i = 0
    for strat_name, risk_col in RISK_COLUMNS.items():
        y = df[risk_col]
        for feat_name, feat_cols in FEATURE_SETS.items():
            X = _encode(df, feat_cols)
            for model_name, factory in MODELS.items():
                for thr_mode in ["default", "youden"]:
                    i += 1
                    t0 = time.time()
                    logging.info("[%d/%d] strat=%s feat=%s model=%s thr=%s",
                                 i, total, strat_name, feat_name,
                                 model_name, thr_mode)
                    folds = run_cv(X, y, factory, threshold_mode=thr_mode,
                                   n_splits=n_splits, random_state=random_state)
                    cfg = ConfigResult(strategy=strat_name, model=model_name,
                                       threshold_mode=thr_mode, features=feat_name,
                                       folds=folds)
                    summary = cfg.summary()
                    summary["elapsed_s"] = round(time.time() - t0, 2)
                    rows.append(summary)
    return pd.DataFrame(rows)


def cross_dataset_sweep(
    df: pd.DataFrame,
    random_state: int = 42,
    seeds: list = None,
    recompute_labels: bool = True,
    include_youden_val: bool = True,
) -> pd.DataFrame:
    """Leave-one-dataset-out sweep across the same grid.

    Addresses review concerns:
    - M1: Multi-seed evaluation for confidence intervals. Pass a list of
      seeds (default: 5 seeds starting from random_state).
    - M3: Recompute std/quartile thresholds on training data only, not the
      pooled dataset. Enabled by default.
    - Q5: Adds a 'youden_val' threshold mode that computes Youden's J on a
      held-out validation split rather than the test set.
    """
    if seeds is None:
        seeds = [random_state + i for i in range(5)]

    thr_modes = ["default", "youden"]
    if include_youden_val:
        thr_modes.append("youden_val")

    rows = []
    total = (len(RISK_COLUMNS) * len(MODELS) * len(thr_modes)
             * len(FEATURE_SETS) * len(seeds))
    i = 0
    for seed in seeds:
        for strat_name, risk_col in RISK_COLUMNS.items():
            for feat_name, feat_cols in FEATURE_SETS.items():
                for model_name, factory in MODELS.items():
                    for thr_mode in thr_modes:
                        i += 1
                        logging.info("[x %d/%d] seed=%d strat=%s feat=%s model=%s thr=%s",
                                     i, total, seed, strat_name, feat_name,
                                     model_name, thr_mode)
                        results = run_cross_dataset(
                            df, feat_cols, risk_col, factory,
                            threshold_mode=thr_mode,
                            random_state=seed,
                            recompute_labels=recompute_labels,
                            strategy_name=strat_name)
                        for r in results:
                            m = r.metrics
                            rows.append({
                                "seed": seed,
                                "strategy": strat_name,
                                "model": model_name,
                                "threshold_mode": thr_mode,
                                "features": feat_name,
                                "train_sources": "+".join(r.train_sources),
                                "test_source": r.test_source,
                                "accuracy": m.accuracy,
                                "precision_weighted": m.precision_weighted,
                                "recall_weighted": m.recall_weighted,
                                "f1_weighted": m.f1_weighted,
                                "recall_very_high": m.recall_very_high,
                                "precision_very_high": m.precision_very_high,
                                "f1_very_high": m.f1_very_high,
                            })
    return pd.DataFrame(rows)


def aggregate_cross_dataset(cross: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed cross-dataset results into mean ± std per config/source."""
    if "seed" not in cross.columns:
        return cross
    grp_cols = ["strategy", "model", "threshold_mode", "features",
                "train_sources", "test_source"]
    metric_cols = ["accuracy", "precision_weighted", "recall_weighted",
                   "f1_weighted", "recall_very_high",
                   "precision_very_high", "f1_very_high"]
    agg = {m: ["mean", "std"] for m in metric_cols}
    out = cross.groupby(grp_cols).agg(agg).reset_index()
    # Flatten MultiIndex columns: leave groupby keys alone, join metric tuples
    new_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            top, sub = c
            if sub == "":
                new_cols.append(top)
            else:
                new_cols.append(f"{top}_{sub}")
        else:
            new_cols.append(c)
    out.columns = new_cols
    return out


def _select_best_config_from_cross(cross: pd.DataFrame):
    """Aggregate multi-seed cross results and return the config with the
    highest *minimum* mean VH recall across held-out sources within the
    'full' feature set. Returns (strategy, model, threshold_mode) or None.
    """
    if cross is None or cross.empty:
        return None
    # Aggregate over seeds (if multi-seed was run)
    grp_cols = ["strategy", "model", "threshold_mode", "features", "test_source"]
    if "seed" in cross.columns:
        mean_by_src = (cross.groupby(grp_cols)["recall_very_high"]
                            .mean().reset_index())
    else:
        mean_by_src = cross.copy()
    # Now take min across test_source per config
    config_cols = ["strategy", "model", "threshold_mode", "features"]
    min_per_config = (mean_by_src.groupby(config_cols)["recall_very_high"]
                                 .min().reset_index())
    full_only = min_per_config[min_per_config["features"] == "full"]
    if full_only.empty:
        full_only = min_per_config
    best = full_only.sort_values("recall_very_high", ascending=False).iloc[0]
    return (best["strategy"], best["model"], best["threshold_mode"],
            float(best["recall_very_high"]))


def best_config_feature_importance(
    df: pd.DataFrame,
    within: pd.DataFrame,
    cross: pd.DataFrame = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pick the best config for feature importance analysis.

    If cross-dataset results are available, pick the config with the highest
    *minimum* VeryHigh recall across held-out datasets — i.e. the most
    consistently transferable configuration.
    Fallback: the within-dataset peak.
    """
    sel = _select_best_config_from_cross(cross)
    if sel is not None:
        strat, model, thr_mode, min_vh = sel
        logging.info("Best config for FI (by min cross-dataset recall_vh): "
                     "strat=%s model=%s thr=%s min_recall_vh=%.3f",
                     strat, model, thr_mode, min_vh)
    else:
        pool = within[within["features"] == "full"].copy()
        if pool.empty:
            return pd.DataFrame()
        best = pool.sort_values("recall_very_high_mean", ascending=False).iloc[0]
        strat, model, thr_mode = best["strategy"], best["model"], best["threshold_mode"]
        logging.info("Best config for FI (within-dataset peak): "
                     "strat=%s model=%s thr=%s recall_vh=%.3f",
                     strat, model, thr_mode, best["recall_very_high_mean"])

    risk_col = RISK_COLUMNS[strat]
    feat_cols = FEATURE_SETS["full"]
    X = _encode(df, feat_cols)
    y = df[risk_col]
    factory = MODELS[model]
    fi = feature_importance(X, y, factory, n_repeats=20, random_state=random_state)
    fi["strategy"] = strat
    fi["model"] = model
    fi["threshold_mode"] = thr_mode
    return fi


def best_config_feature_importance_triangulation(
    df: pd.DataFrame,
    within: pd.DataFrame,
    cross: pd.DataFrame = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Feature importance with three-method triangulation (M4).

    Uses the same config-selection logic as best_config_feature_importance
    but returns permutation-primary + permutation-RF + Gini-RF importances
    for the same feature set, so the user can check whether the
    single-dominant-feature finding replicates across methods.
    """
    from .evaluation import feature_importance_triangulation

    sel = _select_best_config_from_cross(cross)
    if sel is not None:
        strat, model, thr_mode, min_vh = sel
    else:
        pool = within[within["features"] == "full"].copy()
        if pool.empty:
            return pd.DataFrame()
        best = pool.sort_values("recall_very_high_mean", ascending=False).iloc[0]
        strat, model, thr_mode = best["strategy"], best["model"], best["threshold_mode"]

    risk_col = RISK_COLUMNS[strat]
    feat_cols = FEATURE_SETS["full"]
    X = _encode(df, feat_cols)
    y = df[risk_col]
    factory = MODELS[model]
    tri = feature_importance_triangulation(X, y, factory, n_repeats=20,
                                           random_state=random_state)
    tri["strategy"] = strat
    tri["model"] = model
    tri["threshold_mode"] = thr_mode
    return tri


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/harmonized.csv",
                   help="Harmonized CSV from preprocess.py")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--skip-within", action="store_true")
    p.add_argument("--skip-cross", action="store_true")
    p.add_argument("--skip-fi", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--temporal-holdout", action="store_true",
                   help="Run temporal holdout (train pre-cutoff, test post-cutoff).")
    p.add_argument("--cutoff-year", type=int, default=2010,
                   help="Year threshold for temporal holdout split.")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of oversampling seeds for temporal holdout.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_path = Path(args.data)
    if not data_path.exists():
        logging.error("harmonized data not found at %s — run preprocess.py first", data_path)
        return 2
    df = pd.read_csv(data_path)
    logging.info("loaded %d rows from %s", len(df), data_path)
    logging.info("sources: %s", df.groupby("source").size().to_dict())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Initialize so the feature-importance step can reference them regardless
    # of which skip flags the user passed.
    within = None
    cross = None

    # Class distribution sanity check
    for strat, col in RISK_COLUMNS.items():
        counts = df[col].value_counts().to_dict()
        logging.info("label distribution (%s): %s", strat, counts)

    if not args.skip_within:
        logging.info("=== Within-dataset sweep (pooled CV) ===")
        within = within_dataset_sweep(df, n_splits=args.n_splits,
                                      random_state=args.random_state)
        within_path = out / "within_dataset.csv"
        within.to_csv(within_path, index=False)
        logging.info("wrote %s", within_path)

        # Best configs summary
        top = (within.sort_values("recall_very_high_mean", ascending=False)
                     .head(5)[["strategy", "model", "threshold_mode",
                               "features", "accuracy_mean",
                               "recall_very_high_mean", "recall_very_high_std",
                               "f1_very_high_mean"]])
        logging.info("Top 5 configs by VeryHigh recall:\n%s", top.to_string())

    if not args.skip_cross:
        logging.info("=== Leave-one-dataset-out sweep (multi-seed, label-leak-corrected) ===")
        cross = cross_dataset_sweep(df, random_state=args.random_state)
        cross_path = out / "cross_dataset.csv"
        cross.to_csv(cross_path, index=False)
        logging.info("wrote %s (per-seed rows)", cross_path)

        # Aggregated view: mean ± std across seeds (main paper table)
        cross_agg = aggregate_cross_dataset(cross)
        cross_agg_path = out / "cross_dataset_aggregated.csv"
        cross_agg.to_csv(cross_agg_path, index=False)
        logging.info("wrote %s (mean ± std across seeds)", cross_agg_path)

        # Pivot: mean recall_very_high per held-out dataset (over seeds)
        if not cross.empty:
            pivot = (cross.groupby(["test_source", "strategy", "model",
                                    "threshold_mode", "features"])
                          ["recall_very_high"].mean()
                          .unstack("test_source"))
            logging.info("Cross-dataset VeryHigh recall (mean over seeds):\n%s",
                         pivot.to_string())

    if not args.skip_fi and not args.skip_within and within is not None:
        logging.info("=== Feature importance for best config ===")
        # If cross-dataset sweep ran, use its min-across-sources to pick the
        # most transferable config. Otherwise fall back to within-dataset.
        fi = best_config_feature_importance(
            df, within, cross=cross, random_state=args.random_state)
        if not fi.empty:
            fi_path = out / "feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            logging.info("wrote %s", fi_path)
            logging.info("Top 10 features:\n%s", fi.head(10).to_string())

        # --- Triangulation: same config, three importance methods (M4) -----
        logging.info("=== Feature importance triangulation ===")
        tri = best_config_feature_importance_triangulation(
            df, within, cross=cross, random_state=args.random_state)
        if not tri.empty:
            tri_path = out / "feature_importance_triangulation.csv"
            tri.to_csv(tri_path, index=False)
            logging.info("wrote %s", tri_path)
            # Print top-5 per method
            for method in tri["method"].unique():
                sub = (tri[tri["method"] == method]
                       .sort_values("importance_mean", ascending=False)
                       .head(5))
                logging.info("Top 5 (%s):\n%s", method, sub.to_string())
            fi_path = out / "feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            logging.info("wrote %s", fi_path)
            logging.info("Top 10 features:\n%s", fi.head(10).to_string())

    if args.temporal_holdout:
        logging.info("=== Temporal holdout (train pre-%d, test post-%d) ===",
                     args.cutoff_year, args.cutoff_year)

        n_train = (df["year"] < args.cutoff_year).sum()
        n_test  = (df["year"] >= args.cutoff_year).sum()
        logging.info("Split: %d train rows, %d test rows", n_train, n_test)

        if n_train == 0 or n_test == 0:
            logging.error("Empty train or test split — check --cutoff-year.")
        else:
            seeds = [args.random_state + i for i in range(args.n_seeds)]
            temporal = temporal_holdout_sweep(
                df=df,
                cutoff_year=args.cutoff_year,
                seeds=seeds,
                random_state=args.random_state,
                recompute_labels=True,
                include_youden_val=True,
            )

            temporal_path = out / "temporal_holdout.csv"
            temporal.to_csv(temporal_path, index=False)
            logging.info("wrote %s (%d rows)", temporal_path, len(temporal))

            temporal_agg = aggregate_temporal(temporal)
            temporal_agg_path = out / "temporal_holdout_aggregated.csv"
            temporal_agg.to_csv(temporal_agg_path, index=False)
            logging.info("wrote %s", temporal_agg_path)

            if "recall_very_high_mean" in temporal_agg.columns:
                top = (temporal_agg[
                           temporal_agg["test_source"] == "ALL_POST_CUTOFF"]
                       .sort_values("recall_very_high_mean", ascending=False)
                       .head(5)[["strategy", "model", "threshold_mode",
                                 "features", "recall_very_high_mean",
                                 "recall_very_high_std",
                                 "precision_very_high_mean"]])
                logging.info("Top 5 temporal holdout configs:\n%s",
                             top.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())