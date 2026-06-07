"""
evaluation.py

Cross-validated evaluation with Youden's J threshold tuning.

Preserves the per-class Youden's J optimization from the original project
but wraps it in 5-fold stratified cross-validation so reported metrics
include mean and standard deviation instead of single-split numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .classification_strategies import LABELS


# ---------------------------------------------------------------------------
# Youden's J (per-class) — same formulation as the original notebooks
# ---------------------------------------------------------------------------

def youdens_j_per_class(
    y_true_idx: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[int, float]:
    """For each class k, find the threshold on P(class=k) that maximises
    Youden's J = TPR - FPR when treating class k as positive (one-vs-rest).
    Returns a dict {class_index: optimal_threshold}.
    """
    num_classes = y_pred_proba.shape[1]
    thresholds: Dict[int, float] = {}
    for k in range(num_classes):
        binary_true = (y_true_idx == k).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            # Degenerate fold for this class — no variance. Skip.
            thresholds[k] = 0.5
            continue
        fpr, tpr, thr = roc_curve(binary_true, y_pred_proba[:, k])
        j = tpr - fpr
        best = int(np.argmax(j))
        thresholds[k] = float(thr[best])
    return thresholds


def apply_thresholds(y_pred_proba: np.ndarray,
                     thresholds: Dict[int, float]) -> np.ndarray:
    """Adjust predictions using per-class thresholds. Replicates the
    argmax(p >= threshold) logic from the original notebooks.
    """
    n, k = y_pred_proba.shape
    thr_vec = np.array([thresholds[c] for c in range(k)])
    # For each row, pick the class whose prob exceeds its own threshold by
    # the largest margin. Fall back to argmax if none exceed.
    out = np.empty(n, dtype=int)
    margin = y_pred_proba - thr_vec
    for i in range(n):
        row = margin[i]
        exceeded = np.where(row >= 0)[0]
        if len(exceeded) > 0:
            out[i] = exceeded[np.argmax(row[exceeded])]
        else:
            out[i] = int(np.argmax(y_pred_proba[i]))
    return out


# ---------------------------------------------------------------------------
# One CV experiment
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    recall_very_high: float
    precision_very_high: float
    f1_very_high: float


@dataclass
class ConfigResult:
    strategy: str
    model: str
    threshold_mode: str     # "default" or "youden"
    features: str           # short tag describing feature set
    folds: List[FoldResult] = field(default_factory=list)

    def summary(self) -> Dict:
        d = {"strategy": self.strategy, "model": self.model,
             "threshold_mode": self.threshold_mode, "features": self.features}
        metric_names = ["accuracy", "precision_weighted", "recall_weighted",
                        "f1_weighted", "recall_very_high",
                        "precision_very_high", "f1_very_high"]
        for m in metric_names:
            vals = np.array([getattr(f, m) for f in self.folds])
            d[f"{m}_mean"] = float(np.nanmean(vals))
            d[f"{m}_std"] = float(np.nanstd(vals))
        d["n_folds"] = len(self.folds)
        return d


def _fold_metrics(fold_idx: int,
                  y_true_labels: np.ndarray,
                  y_pred_labels: np.ndarray) -> FoldResult:
    # per-class very-high metrics
    vh = "VeryHigh"
    rec_vh = recall_score(y_true_labels, y_pred_labels,
                          labels=[vh], average="macro", zero_division=0)
    pre_vh = precision_score(y_true_labels, y_pred_labels,
                             labels=[vh], average="macro", zero_division=0)
    f1_vh = f1_score(y_true_labels, y_pred_labels,
                     labels=[vh], average="macro", zero_division=0)
    return FoldResult(
        fold=fold_idx,
        accuracy=accuracy_score(y_true_labels, y_pred_labels),
        precision_weighted=precision_score(y_true_labels, y_pred_labels,
                                           average="weighted", zero_division=0),
        recall_weighted=recall_score(y_true_labels, y_pred_labels,
                                     average="weighted", zero_division=0),
        f1_weighted=f1_score(y_true_labels, y_pred_labels,
                             average="weighted", zero_division=0),
        recall_very_high=rec_vh,
        precision_very_high=pre_vh,
        f1_very_high=f1_vh,
    )


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    threshold_mode: str = "default",
    n_splits: int = 5,
    random_state: int = 42,
    oversample: bool = True,
) -> List[FoldResult]:
    """Run stratified K-fold CV. `y` is expected to contain string labels
    from LABELS. Returns one FoldResult per fold.
    """
    # Encode labels to integers in a fixed order so class indices match Youden's J
    le = LabelEncoder()
    le.fit(LABELS)
    y_int = le.transform(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: List[FoldResult] = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y_int)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        # --- Oversample the training set ---
        if oversample:
            try:
                ros = RandomOverSampler(sampling_strategy="auto",
                                        random_state=random_state)
                X_train, y_train = ros.fit_resample(X_train, y_train)
            except ValueError:
                # Happens when a class has only 1 sample in this fold.
                # Fall through without oversampling for this fold.
                pass

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)
        # Ensure columns align with LABELS order by padding if necessary
        if y_pred_proba.shape[1] != len(LABELS):
            # model.classes_ gives the int labels it saw in training
            full = np.zeros((y_pred_proba.shape[0], len(LABELS)))
            for i, c in enumerate(model.classes_):
                full[:, c] = y_pred_proba[:, i]
            y_pred_proba = full

        if threshold_mode == "youden":
            thr = youdens_j_per_class(y_test, y_pred_proba)
            y_pred_int = apply_thresholds(y_pred_proba, thr)
        else:
            y_pred_int = np.argmax(y_pred_proba, axis=1)

        y_true_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred_int)
        results.append(_fold_metrics(fold_i, y_true_labels, y_pred_labels))

    return results


# ---------------------------------------------------------------------------
# Leave-one-dataset-out (cross-dataset generalization — the IRI contribution)
# ---------------------------------------------------------------------------

@dataclass
class CrossDatasetResult:
    strategy: str
    model: str
    threshold_mode: str
    features: str
    train_sources: Tuple[str, ...]
    test_source: str
    metrics: FoldResult   # re-used as a metric container


def run_cross_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_factory,
    threshold_mode: str = "default",
    oversample: bool = True,
    random_state: int = 42,
    recompute_labels: bool = False,
    strategy_name: str = None,
) -> List[CrossDatasetResult]:
    """Leave-one-dataset-out evaluation.

    For each source S: train on (all - S), test on S. This is the core
    'integration' experiment — it measures how well a pipeline learned on
    two datasets generalizes to a third with different definitions and
    reporting practices.

    Parameters
    ----------
    recompute_labels : bool
        If True, recompute std/quartile risk labels using ONLY the training
        set (fixing the label-leakage concern raised in review M3).
        The rule-based strategy is unaffected since its thresholds are fixed.
        Requires `strategy_name` and the `total_victims` column in `df`.
    strategy_name : str
        'rule', 'std', or 'quartile'. Only used when recompute_labels=True.
    threshold_mode : str
        'default', 'youden', or 'youden_val'.
        - 'youden' uses test-set ROC to pick thresholds (the original,
          transductive behavior kept for backward compatibility)
        - 'youden_val' carves 20% of training as a validation split,
          trains on the rest, computes Youden's J on the validation split,
          and applies those thresholds to the test set (fixes review Q5).
    """
    from .classification_strategies import STRATEGIES

    le = LabelEncoder()
    le.fit(LABELS)
    results: List[CrossDatasetResult] = []

    sources = sorted(df["source"].unique().tolist())

    for test_src in sources:
        train_mask = df["source"] != test_src
        test_mask = df["source"] == test_src

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        # ---- Recompute labels on the training set only (fixes M3) -------
        if recompute_labels and strategy_name is not None:
            if strategy_name == "rule":
                # Rule-based has fixed cutoffs; no leakage to fix.
                pass
            else:
                labeler = STRATEGIES[strategy_name](train_df["total_victims"])
                train_df = train_df.assign(**{label_col: train_df["total_victims"].apply(labeler)})
                test_df = test_df.assign(**{label_col: test_df["total_victims"].apply(labeler)})

        X_train_df = train_df[feature_cols]
        X_test_df = test_df[feature_cols]
        y_train = le.transform(train_df[label_col])
        y_test = le.transform(test_df[label_col])

        # One-hot encode on the UNION of categorical values across train+test
        # so the feature matrix shapes match. Only column structure is shared,
        # not label information.
        combined = pd.concat([X_train_df.assign(_split="train"),
                              X_test_df.assign(_split="test")],
                             ignore_index=True)
        cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c != "_split"]
        combined_enc = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
        X_train_enc = combined_enc[combined_enc["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
        X_test_enc = combined_enc[combined_enc["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

        # ---- For validation-split Youden, carve out 20% of training -----
        val_X = val_y = None
        if threshold_mode == "youden_val":
            from sklearn.model_selection import train_test_split
            try:
                X_train_enc, val_X, y_train, val_y = train_test_split(
                    X_train_enc, y_train, test_size=0.2,
                    stratify=y_train, random_state=random_state)
            except ValueError:
                # Stratification fails if some class has <2 members.
                X_train_enc, val_X, y_train, val_y = train_test_split(
                    X_train_enc, y_train, test_size=0.2,
                    random_state=random_state)

        if oversample:
            try:
                ros = RandomOverSampler(sampling_strategy="auto",
                                        random_state=random_state)
                X_train_enc, y_train = ros.fit_resample(X_train_enc, y_train)
            except ValueError:
                pass

        model = model_factory()
        model.fit(X_train_enc, y_train)
        y_pred_proba = model.predict_proba(X_test_enc)
        if y_pred_proba.shape[1] != len(LABELS):
            full = np.zeros((y_pred_proba.shape[0], len(LABELS)))
            for i, c in enumerate(model.classes_):
                full[:, c] = y_pred_proba[:, i]
            y_pred_proba = full

        if threshold_mode == "youden":
            # Transductive: thresholds fit on the test set itself.
            thr = youdens_j_per_class(y_test, y_pred_proba)
            y_pred_int = apply_thresholds(y_pred_proba, thr)
        elif threshold_mode == "youden_val":
            # Fit thresholds on held-out validation split (no test-set access).
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
        metrics = _fold_metrics(0, y_true_labels, y_pred_labels)
        results.append(CrossDatasetResult(
            strategy="",  # filled in by caller
            model="",
            threshold_mode=threshold_mode,
            features="",
            train_sources=tuple(s for s in sources if s != test_src),
            test_source=test_src,
            metrics=metrics,
        ))
    return results


# ---------------------------------------------------------------------------
# Feature importance (permutation-based)
# ---------------------------------------------------------------------------

def feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on a single train/test split (fast, for the paper)."""
    le = LabelEncoder()
    le.fit(LABELS)
    y_int = le.transform(y)

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_int, test_size=0.25, stratify=y_int, random_state=random_state)

    model = model_factory()
    model.fit(X_tr, y_tr)
    imp = permutation_importance(model, X_te, y_te, n_repeats=n_repeats,
                                 random_state=random_state, n_jobs=1)
    return (pd.DataFrame({
                "feature": X.columns,
                "importance_mean": imp.importances_mean,
                "importance_std": imp.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def feature_importance_triangulation(
    X: pd.DataFrame,
    y: pd.Series,
    primary_factory,
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Feature importance triangulation (addresses review concern M4).

    Computes three complementary views of feature importance:
      1. Permutation importance on the primary (user-chosen) model.
      2. Permutation importance on a Random Forest backup model.
      3. Gini importance from the Random Forest backup model.

    Agreement across methods indicates the importance finding is robust;
    disagreement indicates the primary-model result may be method-specific.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    le = LabelEncoder()
    le.fit(LABELS)
    y_int = le.transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_int, test_size=0.25, stratify=y_int, random_state=random_state)

    rows = []

    # --- 1. Permutation importance on the primary model ---
    m1 = primary_factory()
    m1.fit(X_tr, y_tr)
    imp1 = permutation_importance(m1, X_te, y_te, n_repeats=n_repeats,
                                  random_state=random_state, n_jobs=1)
    for i, col in enumerate(X.columns):
        rows.append({
            "feature": col,
            "method": "permutation_primary",
            "importance_mean": imp1.importances_mean[i],
            "importance_std": imp1.importances_std[i],
        })

    # --- 2. Permutation importance on a Random Forest backup ---
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state,
                                class_weight="balanced", max_depth=10)
    rf.fit(X_tr, y_tr)
    imp2 = permutation_importance(rf, X_te, y_te, n_repeats=n_repeats,
                                  random_state=random_state, n_jobs=1)
    for i, col in enumerate(X.columns):
        rows.append({
            "feature": col,
            "method": "permutation_rf",
            "importance_mean": imp2.importances_mean[i],
            "importance_std": imp2.importances_std[i],
        })

    # --- 3. Gini (mean decrease impurity) on the Random Forest ---
    # RF provides mean importance directly; std is computed across trees.
    tree_importances = np.array([tree.feature_importances_ for tree in rf.estimators_])
    for i, col in enumerate(X.columns):
        rows.append({
            "feature": col,
            "method": "gini_rf",
            "importance_mean": float(tree_importances[:, i].mean()),
            "importance_std": float(tree_importances[:, i].std()),
        })

    return pd.DataFrame(rows)