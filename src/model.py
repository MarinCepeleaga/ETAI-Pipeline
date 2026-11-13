"""Model construction."""
from inspect import signature
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectorMixin, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keep features selected by at least ``min_votes`` of three fitted classifiers."""

    def __init__(self, min_votes: int = 2, random_state: Optional[int] = None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        if self.min_votes not in (1, 2, 3):
            raise ValueError(f"min_votes must be 1, 2, or 3, got {self.min_votes}.")

        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("VoteSelector expects a 2D feature matrix and aligned 1D labels.")
        if len(np.unique(y)) != 2:
            raise ValueError("VoteSelector currently supports binary classification only.")

        self.n_features_in_ = X.shape[1]
        forest = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        logistic_params = {
            "C": 0.01,
            "solver": "liblinear",
            "random_state": self.random_state,
        }
        penalty_parameter = signature(LogisticRegression).parameters.get("penalty")
        if penalty_parameter is None or penalty_parameter.default == "deprecated":
            logistic_params["l1_ratio"] = 1.0
        else:
            logistic_params["penalty"] = "l1"
        lasso = LogisticRegression(**logistic_params).fit(X, y)

        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean()).astype(int)
            + (np.abs(lasso.coef_[0]) > 1e-8).astype(int)
        )
        self.support_ = self.votes_ >= self.min_votes
        if not self.support_.any():
            raise ValueError(
                f"VoteSelector selected no features with min_votes={self.min_votes}."
            )
        return self

    def _get_support_mask(self):
        return self.support_


def adaboost(max_depth: int = 1, **params):
    """Build AdaBoost over decision trees of the configured depth."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        **params,
    )


def oblique_random_forest(
    n_estimators: int = 100,
    n_components: int = 10,
    density: Union[float, str] = "auto",
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    **tree_params,
):
    """Build a bagged forest whose trees split on random linear projections."""
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(
    preprocessing_config: dict,
    model_config: dict,
    selection_config: Optional[dict] = None,
    memory=None,
) -> Pipeline:
    """Build a leak-safe preprocessing, selection, and modeling pipeline."""
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if selection_config and selection_config.get("enabled", False):
        method = selection_config.get("method", "vote")
        if method == "vote":
            selector = VoteSelector(
                min_votes=selection_config.get("min_votes", 2),
                random_state=selection_config.get("random_state"),
            )
        elif method == "kbest":
            selector = SelectKBest(
                score_func=f_classif,
                k=selection_config.get("k", 10),
            )
        elif method == "from_model":
            selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=20,
                    random_state=selection_config.get("random_state"),
                    n_jobs=-1,
                ),
                threshold="mean",
            )
        else:
            raise ValueError(
                f"Unknown feature-selection method: {method!r}. "
                "Options: vote, kbest, from_model"
            )
        steps.append(("select", selector))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)
