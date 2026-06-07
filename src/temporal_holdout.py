"""
temporal_holdout.py

Temporal holdout evaluation: train on pre-2010 incidents, test on
post-2010 incidents, using the same pipeline as the LODO protocol
(per-fold label recomputation, random oversampling, Youden's J
thresholding).

This directly addresses the concept-drift concern raised in the paper's
conclusion: does cross-source model performance degrade over time, or
is it stable? Because the split is deterministic by year, the only
source of variance across seeds is the oversampling randomness.

Outputs one row per (seed, strategy, model, threshold_mode, features,
test_source) — compatible with aggregate_cross_dataset() in
run_experiments.py.

Usage (standalone):
    python -m src.temporal_holdout \
        --data data/processed/harmonized.csv \
        --output-dir results/ \
        --cutoff-year 2010
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Lazy import guard: these are resolved when used as a module inside the
# package. If run standalone (python temporal_holdout.py) adjust sys.path
# before importing.
# ---------------------------------------------------------------------------
try:
    from .classification_strategies import LABELS, STRATEGIES
    from .evaluation import _fold_metrics, youdens_j_per_class, apply_thresholds
    from .models import MODELS
except ImportError:
    # Standalone fallback — caller must ensure src/ is on sys.path
    from classification_strategies import LABELS, STRATEGIES   # type: ignore
    from evaluation import _fold_metrics, youdens_j_per_class, apply_thresholds  # type: ignore
    from models import MODELS  # type: ignore


# ---------------------------------------------------------------------------
# Feature sets — mirrors run_experiments.py exactly so results are comparable
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "ctx":  ["incident_area", "open_close", "gender"],
    "full": ["incident_area", "open_close", "gender",
             "age", "mental_health", "race", "multiple_shooters"],
}

RISK_COLUMNS = {
    "rule":     "risk_rule",
    "std":      "risk_std",
    "quartile": "risk_quartile",
}


# ---------------------------------------------------------------------------
# Encoding helper (same logic as run_experiments._encode)
# ---------------------------------------------------------------------------

def _encode_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple:
    """One-hot encode on the UNION of train+test categorical values.

    Returns (X_train_enc, X_test_enc) as float DataFrames with identical
    column structure. No label information crosses the boundary.
    """
    combined = pd.concat(
        [train_df[feature_cols].assign(_split="train"),
         test_df[feature_cols].assign(_split="test")],
        ignore_index=True,
    )
    cat_cols = [c for c in combined.select_dtypes(
        include=["object", "category"]).columns if c != "_split"]
    combined_enc = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    X_train_enc = (combined_enc[combined_enc["_split"] == "train"]
                   .drop(columns=["_split"])
                   .reset_index(drop=True)
                   .astype(float))
    X_test_enc = (combined_enc[combined_enc["_split"] == "test"]
                  .drop(columns=["_split"])
                  .reset_index(drop=True)
                  .astype(float))
    return X_train_enc, X_test_enc


# ---------------------------------------------------------------------------
# Single temporal split
# ---------------------------------------------------------------------------

def run_temporal_holdout(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_factory,
    threshold_mode: str = "default",
    random_state: int = 42,
    recompute_labels: bool = True,
    strategy_name: Optional[str] = None,
    cutoff_year: int = 2010,
) -> List[dict]:
    """Train on year < cutoff_year, test on year >= cutoff_year.

    Returns a list of dicts (one per test_source + overall) with metrics
    compatible with the cross_dataset_sweep output format.
    """
    le = LabelEncoder()
    le.fit(LABELS)

    train_mask = df["year"] < cutoff_year
    test_mask  = df["year"] >= cutoff_year

    if train_mask.sum() == 0:
        raise ValueError(f"No training rows with year < {cutoff_year}.")
    if test_mask.sum() == 0:
        raise ValueError(f"No test rows with year >= {cutoff_year}.")

    train_df = df.loc[train_mask].copy()
    test_df  = df.loc[test_mask].copy()

    # --- Recompute labels on training data only (no leakage) ---
    if recompute_labels and strategy_name is not None and strategy_name != "rule":
        labeler = STRATEGIES[strategy_name](train_df["total_victims"])
        train_df = train_df.assign(
            **{label_col: train_df["total_victims"].apply(labeler)})
        test_df = test_df.assign(
            **{label_col: test_df["total_victims"].apply(labeler)})

    # --- One-hot encode on union of train+test vocabulary ---
    X_train_enc, X_test_enc = _encode_split(train_df, test_df, feature_cols)

    y_train = le.transform(train_df[label_col])
    y_test  = le.transform(test_df[label_col])

    # --- Youden-val: carve validation split from training data ---
    val_X = val_y = None
    if threshold_mode == "youden_val":
        from sklearn.model_selection import train_test_split
        try:
            X_train_enc, val_X, y_train, val_y = train_test_split(
                X_train_enc, y_train,
                test_size=0.2, stratify=y_train,
                random_state=random_state)
        except ValueError:
            X_train_enc, val_X, y_train, val_y = train_test_split(
                X_train_enc, y_train,
                test_size=0.2, random_state=random_state)

    # --- Oversample training set ---
    try:
        ros = RandomOverSampler(sampling_strategy="auto",
                                random_state=random_state)
        X_train_enc, y_train = ros.fit_resample(X_train_enc, y_train)
    except ValueError:
        pass  # Class has <2 samples; skip oversampling for this config

    # --- Fit and predict ---
    model = model_factory()
    model.fit(X_train_enc, y_train)
    y_pred_proba = model.predict_proba(X_test_enc)

    # Pad probability matrix if model did not see all classes in training
    if y_pred_proba.shape[1] != len(LABELS):
        full = np.zeros((y_pred_proba.shape[0], len(LABELS)))
        for i, c in enumerate(model.classes_):
            full[:, c] = y_pred_proba[:, i]
        y_pred_proba = full

    # --- Apply threshold rule ---
    if threshold_mode == "youden":
        thr = youdens_j_per_class(y_test, y_pred_proba)
        y_pred_int = apply_thresholds(y_pred_proba, thr)
    elif threshold_mode == "youden_val" and val_y is not None:
        val_proba = model.predict_proba(val_X)
        if val_proba.shape[1] != len(LABELS):
            full = np.zeros((val_proba.shape[0], len(LABELS)))
            for i, c in enumerate(model.classes_):
                full[:, c] = val_proba[:, i]
            val_proba = full
        thr = youdens_j_per_class(val_y, val_proba)
        y_pred_int = apply_thresholds(y_pred_proba, thr)
    else:
        y_pred_int = np.argmax(y_pred_proba, axis=1)

    y_true_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred_int)

    # --- Overall metrics across all post-cutoff sources ---
    overall_metrics = _fold_metrics(0, y_true_labels, y_pred_labels)

    rows = []
    base = {
        "n_train": int(train_mask.sum()),
        "n_test":  int(test_mask.sum()),
        "cutoff_year": cutoff_year,
    }

    def _metrics_dict(m):
        return {
            "accuracy":            m.accuracy,
            "precision_weighted":  m.precision_weighted,
            "recall_weighted":     m.recall_weighted,
            "f1_weighted":         m.f1_weighted,
            "recall_very_high":    m.recall_very_high,
            "precision_very_high": m.precision_very_high,
            "f1_very_high":        m.f1_very_high,
        }

    rows.append({**base,
                 "test_source": "ALL_POST_CUTOFF",
                 **_metrics_dict(overall_metrics)})

    # --- Per-source breakdown on the test fold ---
    for src in sorted(test_df["source"].unique()):
        src_mask_test = (test_df["source"] == src).values
        if src_mask_test.sum() == 0:
            continue
        src_true = y_true_labels[src_mask_test]
        src_pred = y_pred_labels[src_mask_test]
        src_metrics = _fold_metrics(0, src_true, src_pred)
        rows.append({**base,
                     "test_source": src,
                     "n_test": int(src_mask_test.sum()),
                     **_metrics_dict(src_metrics)})

    return rows


# ---------------------------------------------------------------------------
# Grid sweep — mirrors cross_dataset_sweep() in run_experiments.py
# ---------------------------------------------------------------------------

def temporal_holdout_sweep(
    df: pd.DataFrame,
    cutoff_year: int = 2010,
    seeds: Optional[List[int]] = None,
    random_state: int = 42,
    recompute_labels: bool = True,
    include_youden_val: bool = True,
) -> pd.DataFrame:
    """Run the full strategy × model × threshold × feature-set grid
    under temporal holdout, repeated over multiple seeds.

    The split (year < cutoff vs year >= cutoff) is deterministic;
    seeds only affect oversampling randomness.

    Returns a DataFrame in the same column format as cross_dataset_sweep()
    so aggregate_cross_dataset() can be reused downstream.
    """
    if seeds is None:
        seeds = [random_state + i for i in range(5)]

    thr_modes = ["default", "youden"]
    if include_youden_val:
        thr_modes.append("youden_val")

    rows = []
    total = (len(RISK_COLUMNS) * len(MODELS)
             * len(thr_modes) * len(FEATURE_SETS) * len(seeds))
    i = 0

    for seed in seeds:
        for strat_name, risk_col in RISK_COLUMNS.items():
            for feat_name, feat_cols in FEATURE_SETS.items():
                for model_name, factory in MODELS.items():
                    for thr_mode in thr_modes:
                        i += 1
                        logging.info(
                            "[temporal %d/%d] seed=%d strat=%s feat=%s "
                            "model=%s thr=%s cutoff=%d",
                            i, total, seed, strat_name, feat_name,
                            model_name, thr_mode, cutoff_year)
                        t0 = time.time()
                        try:
                            split_rows = run_temporal_holdout(
                                df=df,
                                feature_cols=feat_cols,
                                label_col=risk_col,
                                model_factory=factory,
                                threshold_mode=thr_mode,
                                random_state=seed,
                                recompute_labels=recompute_labels,
                                strategy_name=strat_name,
                                cutoff_year=cutoff_year,
                            )
                        except ValueError as exc:
                            logging.warning("Skipped: %s", exc)
                            continue

                        for row in split_rows:
                            rows.append({
                                "seed":           seed,
                                "strategy":       strat_name,
                                "model":          model_name,
                                "threshold_mode": thr_mode,
                                "features":       feat_name,
                                "cutoff_year":    cutoff_year,
                                **row,
                                "elapsed_s":      round(time.time() - t0, 3),
                            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: aggregate mean ± std across seeds
# ---------------------------------------------------------------------------

def aggregate_temporal(temporal: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed temporal results into mean ± std per config/source."""
    if "seed" not in temporal.columns or temporal.empty:
        return temporal

    grp_cols = ["strategy", "model", "threshold_mode", "features",
                "cutoff_year", "test_source"]
    metric_cols = ["accuracy", "precision_weighted", "recall_weighted",
                   "f1_weighted", "recall_very_high",
                   "precision_very_high", "f1_very_high"]

    agg = {m: ["mean", "std"] for m in metric_cols}
    out = temporal.groupby(grp_cols).agg(agg).reset_index()

    # Flatten MultiIndex columns
    new_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            top, sub = c
            new_cols.append(top if sub == "" else f"{top}_{sub}")
        else:
            new_cols.append(c)
    out.columns = new_cols
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Temporal holdout evaluation (train pre-cutoff, "
                    "test post-cutoff).")
    p.add_argument("--data", default="data/processed/harmonized.csv")
    p.add_argument("--output-dir", default="results/")
    p.add_argument("--cutoff-year", type=int, default=2010,
                   help="Year threshold: train < cutoff, test >= cutoff.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of oversampling seeds.")
    p.add_argument("--no-youden-val", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_path = Path(args.data)
    if not data_path.exists():
        logging.error("Data not found at %s — run preprocess.py first.",
                      data_path)
        return 2

    df = pd.read_csv(data_path)
    logging.info("Loaded %d rows from %s", len(df), data_path)

    n_train = (df["year"] < args.cutoff_year).sum()
    n_test  = (df["year"] >= args.cutoff_year).sum()
    logging.info("Temporal split: %d train (year < %d), %d test (year >= %d)",
                 n_train, args.cutoff_year, n_test, args.cutoff_year)

    if n_train == 0 or n_test == 0:
        logging.error("Empty train or test split. Check --cutoff-year.")
        return 2

    seeds = [args.random_state + i for i in range(args.n_seeds)]

    temporal = temporal_holdout_sweep(
        df=df,
        cutoff_year=args.cutoff_year,
        seeds=seeds,
        random_state=args.random_state,
        recompute_labels=True,
        include_youden_val=not args.no_youden_val,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / "temporal_holdout.csv"
    temporal.to_csv(raw_path, index=False)
    logging.info("Wrote per-seed results to %s (%d rows)", raw_path, len(temporal))

    agg = aggregate_temporal(temporal)
    agg_path = out / "temporal_holdout_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    logging.info("Wrote aggregated results to %s (%d rows)", agg_path, len(agg))

    # Summary: top 5 configs by VeryHigh recall on ALL_POST_CUTOFF
    if "recall_very_high_mean" in agg.columns:
        top = (agg[agg["test_source"] == "ALL_POST_CUTOFF"]
               .sort_values("recall_very_high_mean", ascending=False)
               .head(5)[["strategy", "model", "threshold_mode", "features",
                          "recall_very_high_mean", "recall_very_high_std",
                          "precision_very_high_mean"]])
        logging.info("Top 5 temporal holdout configs (ALL post-%d):\n%s",
                     args.cutoff_year, top.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
