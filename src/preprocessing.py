"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_placeholders(df: pd.DataFrame, tokens: list) -> pd.DataFrame:
    placeholders = {str(token).strip().casefold() for token in tokens}
    df = df.copy()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        missing = df[column].map(
            lambda value: isinstance(value, str) and value.casefold() in placeholders
        )
        df.loc[missing, column] = pd.NA
    return df


def _clean_numeric_columns(df: pd.DataFrame, columns: set, rules: dict) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_rows = pd.Series(True, index=df.index)
    for column, limits in rules.items():
        if column in df.columns:
            valid_rows &= df[column].between(
                limits.get("min", -float("inf")),
                limits.get("max", float("inf")),
            )
    return df.loc[valid_rows]


def _normalize_categories(df: pd.DataFrame, category_rules: dict) -> pd.DataFrame:
    for column, categories in category_rules.items():
        if column in df.columns:
            canonical = {str(key).strip().casefold(): value for key, value in categories.items()}
            df[column] = df[column].map(
                lambda value: canonical.get(value.casefold(), value)
                if isinstance(value, str)
                else value
            )
    return df


def clean_data(df: pd.DataFrame, diagnostics: dict) -> pd.DataFrame:
    """Clean the dataset using the rules in the ``diagnostics`` config block."""
    df = _normalize_placeholders(df, diagnostics.get("placeholder_tokens", []))

    validity_rules = diagnostics.get("validity_rules", {})
    numeric_columns = set(diagnostics.get("numeric_text_columns", [])) | set(validity_rules)
    df = _clean_numeric_columns(df, numeric_columns, validity_rules)
    df = _normalize_categories(df, diagnostics.get("canonical_categories", {}))

    columns_to_drop = [diagnostics.get("id_column"), *diagnostics.get("redundant_columns", [])]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    out = df.dropna().reset_index(drop=True)
    return out

def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = df.dropna()

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test
