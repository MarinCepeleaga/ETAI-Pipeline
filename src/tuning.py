"""Cross-validation and hyperparameter tuning for the complete model pipeline."""
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from src.evaluate import positive_class_probability

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scorer,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Optimize Pipeline parameters using only the supplied development rows."""
    if not search_space:
        raise ValueError("The tuning search space must define at least one parameter.")
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}.")

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            parameter_type = spec["type"]
            if parameter_type == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif parameter_type == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif parameter_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(
                    f"Unknown search-space type for {name}: {parameter_type}. "
                    "Options: int, float, categorical"
                )

        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=list(cv.split(X, y)),
            scoring=scorer,
            n_jobs=n_jobs,
            error_score="raise",
        )
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv,
    inner_cv,
    scorers: dict,
    selection_metric: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Estimate tuning performance with an outer CV loop independent of tuning."""
    if selection_metric not in scorers:
        raise ValueError(
            f"Selection metric {selection_metric!r} is not present in the scorers."
        )
    rows = []
    oof_proba = np.empty(len(y), dtype=float)

    for fold, (train_indices, validation_indices) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        best_pipeline, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            random_state,
            n_jobs,
        )
        best_pipeline.fit(X_train, y_train)

        train_score = scorers[selection_metric](best_pipeline, X_train, y_train)
        validation_scores = {
            name: scorer(best_pipeline, X_validation, y_validation)
            for name, scorer in scorers.items()
        }
        oof_proba[validation_indices] = positive_class_probability(
            best_pipeline, X_validation
        )
        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": validation_scores[selection_metric],
            "gap": train_score - validation_scores[selection_metric],
            **validation_scores,
            "inner_best": study.best_value,
            **study.best_params,
        })

    return pd.DataFrame(rows), oof_proba


def tuning_report(
    study,
    nested_scores: pd.DataFrame,
    selection_metric: str,
    top: int = 5,
) -> str:
    """Report the final development-set study and its nested-CV optimism check."""
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(
        columns=lambda column: column.replace("params_", "").replace("user_attrs_", "")
    )
    trials = trials.rename(columns={
        "number": "trial",
        "value": f"mean {selection_metric}",
    })
    inner_mean = nested_scores["inner_best"].mean()
    outer_mean = nested_scores["validation"].mean()
    lines = [
        f"Tuning on all development rows ({len(study.trials)} Optuna trials; "
        f"metric: {selection_metric})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f} (optimistic selection score)",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean {selection_metric}", ascending=False)
        .head(top)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "Optimism check (nested CV):",
        f"  inner tuning score mean = {inner_mean:.3f}",
        f"  outer validation score mean = {outer_mean:.3f}",
        f"  optimism = {inner_mean - outer_mean:+.3f} (report the outer validation score)",
    ]
    return "\n".join(lines)
