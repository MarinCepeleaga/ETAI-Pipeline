"""Row-preserving cleaning and leak-safe, configuration-driven preprocessing.

The sensitive attribute stays out of model inputs and is retained only for
the fairness reports.
"""
import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

ENGINEERED_FEATURE_COLUMNS = (
    "priors_per_adult_year",
    "juvenile_offense_total",
    "age_priors_interaction",
)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        lowered = cleaned.map(
            lambda value: value.lower() if isinstance(value, str) else value
        )
        out[col] = lowered.map(mapping).fillna(cleaned)
        placeholders = out[col].map(
            lambda value: isinstance(value, str) and value.strip() in placeholder_tokens
        )
        out.loc[placeholders, col] = np.nan
    return out


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Replace numeric values outside configured bounds with NaN and report counts."""
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """Apply target-agnostic cleaning while preserving every row and its order."""
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].replace(list(placeholder_tokens), np.nan), errors="coerce"
            )

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = _canonicalize_categories(
        out,
        diagnostics_config.get("canonical_categories", {}),
        placeholder_tokens,
    )
    columns_to_drop = [
        col for col in diagnostics_config.get("redundant_columns", []) if col in out.columns
    ]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """Drop duplicate training rows and IDs; do not use on inference data."""
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Add flags for diagnosed missing-not-at-random columns before imputation."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed-formula features that do not learn statistics from the data."""
    required_columns = {
        "age",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot create engineered features; missing columns: {sorted(missing_columns)}"
        )

    out = df.copy()
    out["priors_per_adult_year"] = out["priors_count"] / (out["age"] - 17).clip(lower=1)
    out["juvenile_offense_total"] = out[
        ["juv_fel_count", "juv_misd_count", "juv_other_count"]
    ].sum(axis=1, min_count=3)
    out["age_priors_interaction"] = out["age"] * out["priors_count"]
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """Separate model inputs, optional labels, and fields reserved for fairness audits."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None
    extras_cols = [col for col in [sensitive_attr, "score_text"] if col in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [col for col in df.columns if col not in always_drop]
    return df[feature_cols], y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """Create a stratified development split and a locked test split."""
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test


_SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(
        target_type="binary",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
    ),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """Build an unfitted, config-driven preprocessor for use inside a model Pipeline."""
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{col}_was_missing" for col in indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])
