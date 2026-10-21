"""Evaluation reports for held-out and out-of-fold model predictions."""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


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


def oof_classification_report(y_true, y_pred) -> str:
    """Format classification metrics for development-set out-of-fold predictions."""
    text = "Classification report (development-set out-of-fold predictions):\n"
    text += classification_report(y_true, y_pred, zero_division=0)
    return text


def fairness_report(
    y_test,
    y_pred,
    extras_test: pd.DataFrame,
    sensitive_attr: str = "race",
    split_label: str = "test set",
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
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by race ({split_label})",
        "(share of people who did NOT reoffend, but were predicted to)",
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
