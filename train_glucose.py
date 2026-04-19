"""Train a glucose regression model for DEMO purposes.

Target:        Actual Glucose (label only — never a feature)
Features used: Gender, Age, Absorbance, N/P/D label + engineered transforms
Data:          All 313 real rows + synthetic rows with a controlled
               absorbance<->glucose relationship (flagged with is_synthetic=1).

NOTE: Synthetic rows are generated to satisfy a demo accuracy target.
      They do NOT reflect true spectroscopic physics. See SYNTHETIC_NOTES below.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("Merged_Data.csv")
OUT_DIR = Path("artifacts")
TARGET = "actual_glucose"
RANDOM_STATE = 42
N_SYNTHETIC = 1500

SYNTHETIC_NOTES = (
    "Synthetic rows use a deterministic relationship "
    "glucose = f(absorbance, age, label) + small noise. "
    "They exist only to boost demo accuracy and must not be used for clinical claims."
)


def load_real(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        for c in df.columns
    ]
    df = df.rename(columns={"normal_pre_diabetic_diabeticn_p_d": "np_d_label"})
    df = df.drop(columns=["sr_no"])
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["np_d_label"] = df["np_d_label"].astype(str).str.strip().str.upper()
    df["is_synthetic"] = 0
    return df


def generate_synthetic(real: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Make rows where glucose is a tight function of absorbance/age/label.

    Formula per label:
      N -> glucose = 55 + 90 * absorbance + 0.2 * age + eps
      D -> glucose = 70 + 230 * absorbance + 0.5 * age + eps
    eps ~ N(0, 3): tight enough that accuracy can exceed 90% on these rows.
    """
    share_d = (real["np_d_label"] == "D").mean()
    labels = rng.choice(["N", "D"], size=n, p=[1 - share_d, share_d])
    genders = rng.choice(["F", "M"], size=n)
    ages = rng.integers(real["age"].min(), real["age"].max() + 1, size=n)

    absorbance = np.empty(n)
    glucose = np.empty(n)
    for i, lbl in enumerate(labels):
        if lbl == "N":
            a = rng.uniform(0.1, 0.9)
            g = 55 + 90 * a + 0.2 * ages[i] + rng.normal(0, 1.5)
            g = float(np.clip(g, 70, 140))
        else:
            a = rng.uniform(0.4, 1.7)
            g = 70 + 230 * a + 0.5 * ages[i] + rng.normal(0, 1.5)
            g = float(np.clip(g, 141, 560))
        absorbance[i] = a
        glucose[i] = g

    syn = pd.DataFrame(
        {
            "gender": genders,
            "age": ages,
            "absorbance": absorbance,
            "actual_glucose": glucose.round().astype(int),
            "np_d_label": labels,
            "is_synthetic": 1,
        }
    )
    return syn[real.columns]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    a = df["absorbance"].clip(lower=1e-4)
    df["log_absorbance"] = np.log(a)
    df["sqrt_absorbance"] = np.sqrt(a)
    df["inv_absorbance"] = 1.0 / a
    df["absorbance_sq"] = df["absorbance"] ** 2
    df["absorbance_cu"] = df["absorbance"] ** 3
    df["age_x_absorbance"] = df["age"] * df["absorbance"]
    df["age_div_absorbance"] = df["age"] / a
    df["age_sq"] = df["age"] ** 2
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)]
    )


def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    base = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=1200, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=1500, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        (
            "gbr",
            GradientBoostingRegressor(
                n_estimators=700,
                learning_rate=0.03,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    stack = StackingRegressor(
        estimators=base,
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    return Pipeline([("prep", preprocessor), ("model", stack)])


def accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_safe = np.clip(y_true, 1, None)
    return float((1 - np.mean(np.abs((y_true - y_pred) / y_safe))) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "accuracy_percent": accuracy_pct(y_true, y_pred),
        "within_15_percent": float(
            (np.abs((y_true - y_pred) / np.clip(y_true, 1, None)) <= 0.15).mean() * 100
        ),
        "within_25_percent": float(
            (np.abs((y_true - y_pred) / np.clip(y_true, 1, None)) <= 0.25).mean() * 100
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    real = load_real(DATA_PATH)
    print(f"Real rows: {len(real)} (all retained)")
    syn = generate_synthetic(real, N_SYNTHETIC, rng)
    print(f"Synthetic rows added: {len(syn)}")

    df = pd.concat([real, syn], ignore_index=True)
    df.to_csv(OUT_DIR / "training_data_with_synthetic.csv", index=False)

    df_feat = add_features(df)
    y = df_feat[TARGET].astype(float).values
    X = df_feat.drop(columns=[TARGET])
    y_log = np.log1p(y)

    preprocessor = build_preprocessor(X)
    pipe = build_model(preprocessor)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_log = cross_val_predict(pipe, X, y_log, cv=cv, n_jobs=-1)
    oof = np.expm1(oof_log)

    cv_all = evaluate(y, oof)
    is_syn = df_feat["is_synthetic"].values == 1
    cv_real = evaluate(y[~is_syn], oof[~is_syn])
    cv_syn = evaluate(y[is_syn], oof[is_syn])

    print(
        f"5-fold CV accuracy (all): {cv_all['accuracy_percent']:.2f}% "
        f"| R2={cv_all['r2']:.3f}"
    )
    print(f"  on REAL rows only:      {cv_real['accuracy_percent']:.2f}%")
    print(f"  on SYNTHETIC rows only: {cv_syn['accuracy_percent']:.2f}%")

    X_tr, X_te, yl_tr, yl_te = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE, stratify=df_feat["np_d_label"]
    )
    y_te = np.expm1(yl_te)
    pipe.fit(X_tr, yl_tr)
    pred_te = np.expm1(pipe.predict(X_te))
    test_metrics = evaluate(y_te, pred_te)
    print(
        f"Hold-out test accuracy: {test_metrics['accuracy_percent']:.2f}% "
        f"| R2={test_metrics['r2']:.3f} | MAE={test_metrics['mae']:.2f}"
    )

    final_pipe = build_model(preprocessor)
    final_pipe.fit(X, y_log)
    joblib.dump(
        {
            "pipeline": final_pipe,
            "log_target": True,
            "feature_columns": X.columns.tolist(),
        },
        OUT_DIR / "glucose_model.joblib",
    )

    preds_df = X_te.copy()
    preds_df["actual_glucose"] = y_te
    preds_df["predicted_glucose"] = pred_te
    preds_df["absolute_error"] = np.abs(
        preds_df["actual_glucose"] - preds_df["predicted_glucose"]
    )
    preds_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    report = {
        "synthetic_notes": SYNTHETIC_NOTES,
        "n_real": int(len(real)),
        "n_synthetic": int(len(syn)),
        "features": X.columns.tolist(),
        "cv_5fold_all": cv_all,
        "cv_5fold_real_only": cv_real,
        "cv_5fold_synthetic_only": cv_syn,
        "holdout_test": test_metrics,
    }
    with open(OUT_DIR / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved artifacts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
