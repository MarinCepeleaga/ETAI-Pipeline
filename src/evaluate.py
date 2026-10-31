"""Evaluation reports for held-out and out-of-fold model predictions."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)
from sklearn.model_selection import cross_validate


def positive_class_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities for the binary positive class (label 1)."""
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Positive-class probabilities require class label 1.")
    return estimator.predict_proba(X)[:, classes.index(1)]


def build_scorers(metrics: list) -> dict:
    """Build scikit-learn scorers, including configurable F-beta metrics."""
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            if beta <= 0:
                raise ValueError(f"F-beta requires beta > 0, got {beta}.")
            scorers[name] = make_scorer(
                fbeta_score,
                beta=beta,
                zero_division=0,
            )
        else:
            scorers[name] = get_scorer(name)
    return scorers


def _predictions_at_threshold(y_proba, threshold: float) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise ValueError(f"Probability threshold must be between 0 and 1, got {threshold}.")
    probabilities = np.asarray(y_proba)
    if probabilities.ndim != 1:
        raise ValueError("Positive-class probabilities must be a one-dimensional array.")
    return (probabilities >= threshold).astype(int)


def cross_validate_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scorers: dict,
    selection_metric: str,
    n_jobs: int = 1,
):
    """Score each fold on every metric and return leak-safe OOF probabilities."""
    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric {selection_metric!r} is not present in the scorers."
        )

    splits = list(cv.split(X, y))
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=splits,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
    fold_scores = pd.DataFrame({"fold": np.arange(1, len(splits) + 1)})
    fold_scores["train"] = results[f"train_{selection_metric}"]
    fold_scores["validation"] = results[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = results[f"test_{name}"]

    oof_proba = np.empty(len(X), dtype=float)
    for (_, validation_indices), estimator in zip(splits, results["estimator"]):
        oof_proba[validation_indices] = positive_class_probability(
            estimator, X.iloc[validation_indices]
        )
    return fold_scores, oof_proba


def cv_report(
    fold_scores: pd.DataFrame,
    metrics: list,
    selection_metric: str,
    title: str = "Cross-validation",
) -> str:
    """Format per-fold and mean/std results for all configured metrics."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds; "
        f"selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        fold_scores[metrics]
        .agg(["mean", "std"])
        .T.to_string(float_format=lambda value: f"{value:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = "
        f"{fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def evaluate(y_train, y_train_pred, y_test, y_pred) -> str:
    """Return train/test accuracy and the test-set classification report."""
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    gap = train_accuracy - test_accuracy

    lines = [
        f"Train accuracy: {train_accuracy:.3f}",
        f"Test accuracy:  {test_accuracy:.3f}",
        f"Gap (train - test): {gap:+.3f}",
    ]
    lines.append("")
    lines.append("Classification report (test set):")
    lines.append(classification_report(y_test, y_pred, zero_division=0))

    return "\n".join(lines)


def oof_classification_report(y_true, y_proba, threshold: float = 0.5) -> str:
    """Report thresholded OOF classification metrics and their confusion matrix."""
    y_pred = _predictions_at_threshold(y_proba, threshold)
    text = (
        "Classification report (development-set out-of-fold predictions; "
        f"threshold={threshold:.2f}):\n"
    )
    text += classification_report(y_true, y_pred, zero_division=0)
    text += "\nConfusion matrix (rows=true, columns=predicted; labels 0, 1):\n"
    text += np.array2string(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    return text


def fairness_report(
    y_test,
    y_proba,
    extras_test: pd.DataFrame,
    sensitive_attr: str = "race",
    split_label: str = "test set",
    threshold: float = 0.5,
) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each race group, prints (and returns as text) the false
    positive rate (share of people who did NOT reoffend but were
    predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras_test.copy()
    df["y_true"] = y_test.values
    df["y_pred_model"] = _predictions_at_threshold(y_proba, threshold)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by race ({split_label})",
        "(share of people who did NOT reoffend, but were predicted to)",
        f"(model probability threshold = {threshold:.2f})",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    return "\n".join(lines)
