"""
classification_strategies.py

Risk-level labeling strategies. Preserves the three strategies from the
original project (rule-based, standard-deviation, quartile).

These strategies are ALSO applied in preprocess.py on the pooled dataset.
This module exposes the same functions for per-fold recomputation if needed
(e.g., computing std/quartile bounds only on the training split to avoid
label leakage — see evaluation.py).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

LABELS = ["Low", "Medium", "High", "VeryHigh"]


def rule_based_labeler() -> Callable[[float], str]:
    """Fixed thresholds from the original project."""
    def f(x: float) -> str:
        if x < 10:
            return "Low"
        if 11 <= x <= 20:
            return "Medium"
        if 21 <= x <= 40:
            return "High"
        return "VeryHigh"
    return f


def std_labeler(victims: pd.Series) -> Callable[[float], str]:
    mu, sd = victims.mean(), victims.std()

    def f(x: float) -> str:
        if x < mu - sd:
            return "Low"
        if x <= mu + sd:
            return "Medium"
        if x <= mu + 2 * sd:
            return "High"
        return "VeryHigh"
    return f


def quartile_labeler(victims: pd.Series) -> Callable[[float], str]:
    q1, q2, q3 = victims.quantile([0.25, 0.5, 0.75])

    def f(x: float) -> str:
        if x <= q1:
            return "Low"
        if x <= q2:
            return "Medium"
        if x <= q3:
            return "High"
        return "VeryHigh"
    return f


STRATEGIES: Dict[str, Callable[[pd.Series], Callable[[float], str]]] = {
    # rule-based ignores the data distribution; wrap for consistent API
    "rule":     lambda victims: rule_based_labeler(),
    "std":      std_labeler,
    "quartile": quartile_labeler,
}


def label_series(strategy: str, victims: pd.Series) -> pd.Series:
    """Apply a strategy to produce a Series of risk labels."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {list(STRATEGIES)}")
    labeler = STRATEGIES[strategy](victims)
    return victims.apply(labeler)
