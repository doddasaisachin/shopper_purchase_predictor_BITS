"""
Train five classifiers on the UCI Online Shoppers Purchasing Intention dataset,
evaluate them on a held-out test split and persist everything the Streamlit
app needs (fitted pipelines, metrics table, test CSV).

Run from the project root:
    python model/train_models.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "Revenue"
NUMERIC_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
# Integer-coded columns with no ordinal meaning are treated as categories.
CATEGORICAL_COLS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Weekend"] = df["Weekend"].astype(str).str.upper()
    df[TARGET] = df[TARGET].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def candidate_models():
    """Each entry: (estimator, small hyper-parameter grid tuned with CV on train only)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 20, 50]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
        ),
        "Support Vector Machine": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
            {"clf__C": [0.5, 1.0, 5.0]},
        ),
    }


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    # Labels come from predict_proba (not predict) so every model, including SVC,
    # uses the same rule as the Streamlit app's threshold slider.
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_dataset()
    X, y = df[FEATURE_COLS], df[TARGET]
    print(f"Dataset: {X.shape[0]} rows x {X.shape[1]} features | positive rate = {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    test_df = X_test.copy()
    test_df[TARGET] = y_test.map({1: "TRUE", 0: "FALSE"})
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved held-out test set -> {TEST_CSV.name} ({len(test_df)} rows)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        metrics = evaluate(best, X_test, y_test)
        rows.append({"Model": name, **metrics})
        print(f"{name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    metrics_df = pd.DataFrame(rows).set_index("Model").round(4)
    metrics_df.to_csv(MODEL_DIR / "metrics.csv")

    meta = {
        "target": TARGET,
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "model_files": MODEL_FILES,
        "best_params": best_params,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": SEED,
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n" + metrics_df.to_string())


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
