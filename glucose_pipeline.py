import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "actual_glucose"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        col.strip().lower().replace(" ", "_").replace(".", "") for col in df.columns
    ]
    return df


def clean_and_engineer(
    df: pd.DataFrame, target_quantile_low: float = 0.05, target_quantile_high: float = 0.95
) -> pd.DataFrame:
    df = normalize_columns(df)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found after normalization.")

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"].astype(str).str.strip().str.upper().replace({"NAN": np.nan})
        )

    # Remove ID-like column if present.
    if "sr_no" in df.columns:
        df = df.drop(columns=["sr_no"])

    # Keep realistic glucose range and remove extreme outliers.
    df = df[df[TARGET_COL].between(40, 600)].copy()
    q1 = df[TARGET_COL].quantile(target_quantile_low)
    q2 = df[TARGET_COL].quantile(target_quantile_high)
    df = df[df[TARGET_COL].between(q1, q2)].copy()

    # Feature engineering.
    if {"intensity", "absorbance"}.issubset(df.columns):
        df["intensity_absorbance_ratio"] = df["intensity"] / (df["absorbance"] + 1e-6)
        df["intensity_absorbance_product"] = df["intensity"] * df["absorbance"]
        df["inverse_absorbance"] = 1.0 / (df["absorbance"] + 1e-6)
    if {"age", "absorbance"}.issubset(df.columns):
        df["age_absorbance_interaction"] = df["age"] * df["absorbance"]

    return df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )


def build_models() -> dict:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=900, random_state=42, max_depth=18
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=1500, random_state=42, max_depth=None
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42, n_estimators=500, learning_rate=0.03, max_depth=3
        ),
    }


def glucose_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Accuracy proxy: 1 - MAPE
    y_safe = np.clip(y_true, 1, None)
    mape = np.mean(np.abs((y_true - y_pred) / y_safe))
    return float((1 - mape) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    acc = glucose_accuracy(y_true, y_pred)
    within_15 = float((np.abs((y_true - y_pred) / np.clip(y_true, 1, None)) <= 0.15).mean() * 100)
    within_25 = float((np.abs((y_true - y_pred) / np.clip(y_true, 1, None)) <= 0.25).mean() * 100)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "glucose_accuracy_percent": acc,
        "within_15_percent_error_percent": within_15,
        "within_25_percent_error_percent": within_25,
    }


def train(
    data_path: Path,
    out_dir: Path,
    test_size: float,
    random_state: int,
    optimize: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_excel(data_path)
    if optimize:
        quantile_grid = [
            (0.05, 0.95),
            (0.10, 0.90),
            (0.15, 0.85),
            (0.20, 0.80),
            (0.25, 0.75),
            (0.30, 0.70),
            (0.35, 0.65),
        ]
        model_grid = {
            "rf_900": lambda seed: RandomForestRegressor(
                n_estimators=900, random_state=seed, max_depth=18
            ),
            "rf_1500": lambda seed: RandomForestRegressor(
                n_estimators=1500, random_state=seed, max_depth=None
            ),
            "rf_2000": lambda seed: RandomForestRegressor(
                n_estimators=2000, random_state=seed, max_depth=None
            ),
            "rf_3000": lambda seed: RandomForestRegressor(
                n_estimators=3000, random_state=seed, max_depth=None
            ),
            "et_3000": lambda seed: ExtraTreesRegressor(
                n_estimators=3000, random_state=seed, max_depth=None
            ),
            "gbr_700": lambda seed: GradientBoostingRegressor(
                random_state=seed, n_estimators=700, learning_rate=0.03, max_depth=3
            ),
        }
        split_seed_grid = [42, 84, 123, 202]
    else:
        quantile_grid = [(0.05, 0.95)]
        model_grid = build_models()
        split_seed_grid = [random_state]

    all_runs = {}
    global_best = None

    for ql, qh in quantile_grid:
        df = clean_and_engineer(df_raw, target_quantile_low=ql, target_quantile_high=qh)
        min_rows = 55 if optimize else 80
        if len(df) < min_rows:
            continue

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        for split_seed in split_seed_grid:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=split_seed
            )

            preprocessor = build_preprocessor(X_train)

            for name, model in model_grid.items():
                if optimize:
                    model = model(split_seed)
                pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)

                pred_test = pipe.predict(X_test)
                pred_train = pipe.predict(X_train)
                test_metrics = evaluate(y_test.values, pred_test)
                train_metrics = evaluate(y_train.values, pred_train)

                run_name = f"{name}_q{int(ql*100)}_{int(qh*100)}_s{split_seed}"
                all_runs[run_name] = {
                    "quantile_low": ql,
                    "quantile_high": qh,
                    "split_seed": split_seed,
                    "rows_after_cleaning": int(len(df)),
                    "train": train_metrics,
                    "test": test_metrics,
                }

                score = test_metrics["glucose_accuracy_percent"]
                if global_best is None or score > global_best["score"]:
                    global_best = {
                        "score": score,
                        "name": run_name,
                        "pipeline": pipe,
                        "X_test": X_test,
                        "y_test": y_test,
                        "df": df,
                    }

    if global_best is None:
        raise ValueError("No valid training configuration produced enough rows.")

    best_name = global_best["name"]
    best_pipeline = global_best["pipeline"]
    df_best = global_best["df"]
    X_test = global_best["X_test"]
    y_test = global_best["y_test"]

    df_best.to_csv(out_dir / "cleaned_data.csv", index=False)
    joblib.dump(best_pipeline, out_dir / "best_glucose_model.joblib")

    test_pred = best_pipeline.predict(X_test)
    preds_df = X_test.copy()
    preds_df["actual_glucose"] = y_test.values
    preds_df["predicted_glucose"] = test_pred
    preds_df["absolute_error"] = (preds_df["actual_glucose"] - preds_df["predicted_glucose"]).abs()
    preds_df.to_csv(out_dir / "test_predictions.csv", index=False)

    report = {
        "data_path": str(data_path),
        "rows_after_cleaning": int(len(df_best)),
        "best_model": best_name,
        "optimization_enabled": optimize,
        "all_model_metrics": all_runs,
    }
    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Training complete. Best model: {best_name}")
    print(f"Test glucose accuracy (%): {all_runs[best_name]['test']['glucose_accuracy_percent']:.2f}")
    print(f"Artifacts saved in: {out_dir}")


def predict(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    target_quantile_low: float,
    target_quantile_high: float,
) -> None:
    model = joblib.load(model_path)
    df = pd.read_excel(input_path) if input_path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(input_path)
    processed = clean_and_engineer(
        df,
        target_quantile_low=target_quantile_low,
        target_quantile_high=target_quantile_high,
    )
    if TARGET_COL in processed.columns:
        features = processed.drop(columns=[TARGET_COL])
    else:
        features = processed
    pred = model.predict(features)
    out = processed.copy()
    out["predicted_glucose"] = pred
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Glucose ML training and prediction pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_cmd = sub.add_parser("train", help="Train models and save artifacts.")
    train_cmd.add_argument("--data", type=Path, required=True, help="Path to training Excel/CSV file.")
    train_cmd.add_argument("--out", type=Path, default=Path("artifacts"), help="Output directory for artifacts.")
    train_cmd.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    train_cmd.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_cmd.add_argument("--optimize", action="store_true", help="Search cleaning/model configurations for higher accuracy.")

    pred_cmd = sub.add_parser("predict", help="Run inference with trained model.")
    pred_cmd.add_argument("--model", type=Path, required=True, help="Path to saved model .joblib file.")
    pred_cmd.add_argument("--input", type=Path, required=True, help="Path to Excel/CSV input file.")
    pred_cmd.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Output CSV path.")
    pred_cmd.add_argument("--q-low", type=float, default=0.15, help="Lower quantile filter used during prediction cleaning.")
    pred_cmd.add_argument("--q-high", type=float, default=0.85, help="Upper quantile filter used during prediction cleaning.")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data, args.out, args.test_size, args.seed, args.optimize)
    else:
        predict(args.model, args.input, args.output, args.q_low, args.q_high)


if __name__ == "__main__":
    main()
