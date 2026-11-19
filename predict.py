"""Score new CSV rows with a fitted pipeline saved by main.py."""
import argparse

import joblib
import pandas as pd

from src.data import load_config, load_data
from src.evaluate import positive_class_probability
from src.preprocessing import add_engineered_features, clean_dataset, split_features_target

DEFAULT_MODEL_PATH = "models/model.joblib"


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Return IDs, positive-class probabilities, and predictions in input order."""
    diagnostics = config["diagnostics"]
    id_column = diagnostics.get("id_column") or config["data"].get("id_column", "id")
    if id_column not in df.columns:
        raise ValueError(f"Input CSV is missing the ID column {id_column!r}.")

    clean = clean_dataset(df, diagnostics)
    if config.get("feature_engineering", {}).get("enabled", True):
        clean = add_engineered_features(clean)
    X, _, _ = split_features_target(
        clean, config["data"], config["preprocessing"].get("mnar_indicator_sources", [])
    )
    probability = positive_class_probability(pipeline, X)
    return pd.DataFrame({
        id_column: df[id_column],
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with the model's input columns")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="output CSV")
    parser.add_argument("--config", default="config.yaml", help="config used to train the model")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        model_path = config.get("output", {}).get("model_path", DEFAULT_MODEL_PATH)
        pipeline = joblib.load(model_path)
        predictions = predict(load_data(args.input), pipeline, config)
        predictions.to_csv(args.output, index=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{len(predictions)} rows predicted -> {args.output}")


if __name__ == "__main__":
    main()
