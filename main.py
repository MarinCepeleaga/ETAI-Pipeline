"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> cross-validate/tune -> evaluate -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    ENGINEERED_FEATURE_COLUMNS,
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_pipeline
from src.evaluate import evaluate, fairness_report, oof_classification_report
from src.results import save_run
from src.tuning import (
    cross_validate_pipeline,
    cv_report,
    nested_cross_validate,
    tune_pipeline,
    tuning_report,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_train = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    feature_engineering_enabled = config.get("feature_engineering", {}).get(
        "enabled", True
    )
    if feature_engineering_enabled:
        df_train = add_engineered_features(df_train)

    preprocessing_config = dict(config["preprocessing"])
    preprocessing_config["numeric_features"] = list(
        preprocessing_config["numeric_features"]
    )
    if not feature_engineering_enabled:
        preprocessing_config["numeric_features"] = [
            column
            for column in preprocessing_config["numeric_features"]
            if column not in ENGINEERED_FEATURE_COLUMNS
        ]

    X, y, extras = split_features_target(
        df_train,
        config["data"],
        preprocessing_config.get("mnar_indicator_sources", []),
    )
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    scoring = tuning_config.get("scoring", "accuracy")
    n_splits = tuning_config.get("n_splits", 5)
    tuning_seed = tuning_config.get(
        "random_state", config["test_set"]["random_state"]
    )
    n_jobs = tuning_config.get("n_jobs", 1)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=tuning_seed,
    )
    inner_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=tuning_seed,
    )
    pipeline = build_pipeline(preprocessing_config, config["model"])
    report_sections = []

    default_scores, y_oof = cross_validate_pipeline(
        pipeline,
        X_train,
        y_train,
        cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    report_sections.append(cv_report(default_scores, scoring))

    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces", {})
        if model_type not in search_spaces:
            raise ValueError(
                f"Tuning is enabled, but no search space is configured for "
                f"model type {model_type!r}."
            )
        search_space = search_spaces[model_type]
        n_trials = tuning_config.get("n_trials", 30)

        nested_scores, y_oof = nested_cross_validate(
            pipeline,
            X_train,
            y_train,
            cv,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        report_sections.append(cv_report(nested_scores, scoring))

        model, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scoring,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        report_sections.append(tuning_report(study, nested_scores, scoring))
    else:
        model = pipeline

    model.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    report_sections.append(oof_classification_report(y_train, y_oof))
    report_sections.append(
        fairness_report(
            y_train,
            y_oof,
            extras_train,
            sensitive_attr=config["data"]["sensitive_attr"],
            split_label="development-set out-of-fold predictions",
        )
    )
    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report_sections.append(report)
    report_sections.append(
        fairness_report(
            y_test,
            y_test_pred,
            extras_test,
            sensitive_attr=config["data"]["sensitive_attr"],
            split_label="locked test set",
        )
    )
    report = "\n\n".join(report_sections)

    print(report)
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
