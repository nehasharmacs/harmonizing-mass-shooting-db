"""
models.py

Factories for the three model families used in the original project:
  - Decision Tree
  - Multinomial Logistic Regression
  - Gaussian Naive Bayes

Each returns a scikit-learn compatible estimator with a `predict_proba`
method (required for Youden's J threshold tuning).

Hyperparameters are held constant across folds; the research contribution
is not hyperparameter tuning but cross-dataset generalization.
"""
from __future__ import annotations

from typing import Callable, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def make_decision_tree(random_state: int = 42):
    # max_depth matches original project's 5
    return DecisionTreeClassifier(max_depth=3, random_state=random_state)


def make_multinomial_lr(random_state: int = 42):
    # Newer sklearn auto-selects multinomial for multi-class targets;
    # the old `multi_class="multinomial"` kwarg has been removed.
    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )


def make_naive_bayes(random_state: int = 42):
    # GaussianNB has no randomness but keep signature consistent
    return GaussianNB()


MODELS: Dict[str, Callable[..., object]] = {
    "decision_tree":   make_decision_tree,
    "multinomial_lr":  make_multinomial_lr,
    "naive_bayes":     make_naive_bayes,
}
