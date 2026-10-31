"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> cross-validate/tune -> evaluate -> save results
"""
from datetime import datetime

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
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    evaluate,
    fairness_report,
    oof_classification_report,
    positive_class_probability,
)
from src.results import save_run, save_scores
from src.tuning import (
    nested_cross_validate,
    tune_pipeline,
    tuning_report,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    cv_config = config.get("cv", {})
    selection_metric = cv_config.get(
        "selection_metric", tuning_config.get("scoring", "accuracy")
    )
    metrics = cv_config.get("metrics", [selection_metric])
    if selection_metric not in metrics:
        raise ValueError(
            f"Selection metric {selection_metric!r} must be included in cv.metrics."
        )
    scorers = build_scorers(metrics)
    n_splits = cv_config.get("n_splits", tuning_config.get("n_splits", 5))
    tuning_seed = cv_config.get(
        "random_state",
        tuning_config.get("random_state", config["test_set"]["random_state"]),
    )
    n_jobs = cv_config.get("n_jobs", tuning_config.get("n_jobs", 1))
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=cv_config.get("shuffle", True),
        random_state=tuning_seed if cv_config.get("shuffle", True) else None,
    )
    inner_splits = tuning_config.get("n_splits", n_splits)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=tuning_seed)
    pipeline = build_pipeline(preprocessing_config, config["model"])
    report_sections = []

    floor_pipeline = build_pipeline(preprocessing_config, {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_train,
        y_train,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs=n_jobs,
    )
    floor_score = float(floor_scores["validation"].mean())

    default_scores, y_oof = cross_validate_pipeline(
        pipeline,
        X_train,
        y_train,
        cv,
        scorers,
        selection_metric,
        n_jobs=n_jobs,
    )
    report_sections.append(
        f"Model: {config['model']['type']}\n"
        f"Selection metric: {selection_metric}\n"
        f"No-model floor (dummy, same folds): {selection_metric} = {floor_score:.3f}"
    )
    report_sections.append(
        cv_report(default_scores, metrics, selection_metric, "Default hyperparameters")
    )
    final_scores = default_scores

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
            scorers,
            selection_metric,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        final_scores = nested_scores
        report_sections.append(
            cv_report(
                final_scores,
                metrics,
                selection_metric,
                "Nested cross-validation (tuning procedure)",
            )
        )

        model, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        report_sections.append(tuning_report(study, final_scores, selection_metric))
    else:
        model = pipeline

    report_sections.append(
        f"All metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, "
        "mean +/- std):\n"
        + "\n".join(
            f"  {name:<18s} {final_scores[name].mean():.3f} +/- "
            f"{final_scores[name].std(ddof=1):.3f}"
            for name in metrics
        )
    )
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

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report_sections.append(report)
    report_sections.append(
        fairness_report(
            y_test,
            positive_class_probability(model, X_test),
            extras_test,
            sensitive_attr=config["data"]["sensitive_attr"],
            split_label="locked test set",
        )
    )
    report = "\n\n".join(report_sections)

    row = {
        "run": run_id,
        "data": config["data"]["path"],
        "model": config["model"]["type"],
        "tuned": tuning_enabled,
        "encoder": preprocessing_config["encoder"],
        "scaler": preprocessing_config["scaler"],
        "params": str(config["model"].get("params", {})),
        "tuned_params": str(study.best_params) if tuning_enabled else "",
        "metric": selection_metric,
        "mean": float(final_scores["validation"].mean()),
        "std": float(final_scores["validation"].std(ddof=1)),
        "untuned": float(default_scores["validation"].mean()),
        "floor": floor_score,
        **{
            f"fold_{fold}": float(score)
            for fold, score in enumerate(final_scores["validation"], start=1)
        },
    }
    scoreboard_path = config.get("output", {}).get(
        "scoreboard", "results/runs.csv"
    )
    runs = save_scores(scoreboard_path, row)
    comparable_runs = runs[
        (runs["data"] == row["data"]) & (runs["metric"] == selection_metric)
    ]
    scoreboard = (
        comparable_runs.sort_values("mean", ascending=False)[
            [
                "run",
                "model",
                "tuned",
                "encoder",
                "scaler",
                "mean",
                "std",
                "untuned",
                "floor",
            ]
        ]
        .rename(columns={"mean": selection_metric})
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    report += f"\n\nScoreboard (same data and selection metric):\n{scoreboard}\n"

    print(report)
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report, run_id=run_id)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
