"""
Shopper Purchase Predictor - Streamlit front-end.

Loads the five pre-trained pipelines from ./model and evaluates them on a
user-uploaded test CSV. No training happens here.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    train_metrics = pd.read_csv(MODEL_DIR / "metrics.csv", index_col="Model")
    return meta, models, train_metrics


def normalise_target(series: pd.Series) -> pd.Series:
    mapping = {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "YES": 1, "NO": 0}
    return series.astype(str).str.strip().str.upper().map(mapping)


def prepare_input(df: pd.DataFrame, meta: dict):
    target = meta["target"]
    features = meta["numeric_features"] + meta["categorical_features"]
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        return None, None, missing

    X = df[features].copy()
    X["Weekend"] = X["Weekend"].astype(str).str.upper()
    for col in meta["categorical_features"]:
        X[col] = X[col].astype(str)
    y = normalise_target(df[target])

    valid = y.notna()
    return X[valid], y[valid].astype(int), []


def score(model, X, y, threshold: float = 0.5) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    has_both = y.nunique() == 2
    return {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if has_both else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
LABELS = ["No Purchase", "Purchase"]

meta, models, train_metrics = load_artifacts()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)
    st.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))
    threshold = st.slider("Decision threshold (P(purchase))", 0.05, 0.95, 0.50, 0.05)

    with st.expander("Tuned hyper-parameters"):
        st.json(meta["best_params"].get(model_name, {}))

# ---------------- Header ----------------
st.title("🛒 Shopper Purchase Predictor")
st.caption(
    "Will an online shopping session end in a purchase? Five classifiers trained on the "
    "UCI *Online Shoppers Purchasing Intention* dataset, evaluated live on your test data."
)

if uploaded is not None and not use_sample:
    raw_df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    raw_df = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a test CSV from the sidebar, or tick **Use bundled test_data.csv**.")
    st.stop()

X, y, missing = prepare_input(raw_df, meta)
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()
if len(y) == 0:
    st.error("No usable rows: the `Revenue` column must contain TRUE/FALSE or 1/0 values.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Source", source)
c2.metric("Rows evaluated", f"{len(y):,}")
c3.metric("Purchase rate", f"{y.mean():.1%}")

tab_model, tab_compare, tab_preds, tab_data = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔮 Predictions", "🗂️ Data preview"]
)

# ---------------- Selected model ----------------
with tab_model:
    res = score(models[model_name], X, y, threshold)
    y_prob, y_pred = res["y_prob"], res["y_pred"]

    st.subheader(f"{model_name} — threshold {threshold:.2f}")
    cols = st.columns(len(METRIC_KEYS))
    for col, key in zip(cols, METRIC_KEYS):
        col.metric(key, "n/a" if np.isnan(res[key]) else f"{res[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:
        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, y_prob)
            ax.plot(fpr, tpr, color="#2e7d32", lw=2, label=f"AUC = {res['AUC']:.3f}")
            ax.legend(loc="lower right")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("**Classification report**")
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=LABELS, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---------------- Compare all ----------------
with tab_compare:
    st.subheader("All five models on the uploaded data (threshold 0.50)")
    rows = []
    for name, mdl in models.items():
        r = score(mdl, X, y)
        rows.append({"Model": name, **{k: r[k] for k in METRIC_KEYS}})
    live = pd.DataFrame(rows).set_index("Model").round(4)

    st.dataframe(
        live.style.highlight_max(axis=0, color="#c8e6c9").format("{:.4f}"),
        width="stretch",
    )

    best = live["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** (MCC = {live.loc[best, 'MCC']:.4f})")

    metric_to_plot = st.radio("Metric to chart", METRIC_KEYS, horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    order = live[metric_to_plot].sort_values()
    ax.barh(order.index, order.values,
            color=["#2e7d32" if m == order.idxmax() else "#a5d6a7" for m in order.index])
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_to_plot)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    with st.expander("Reference: metrics recorded at training time (held-out split)"):
        st.dataframe(train_metrics, width="stretch")

# ---------------- Predictions ----------------
with tab_preds:
    st.subheader(f"Row-level predictions — {model_name}")
    out = raw_df.loc[X.index].copy()
    out["P(purchase)"] = y_prob.round(4)
    out["Predicted"] = np.where(y_pred == 1, "TRUE", "FALSE")
    out["Correct"] = y_pred == y.values

    only_errors = st.checkbox("Show misclassified rows only")
    view = out[~out["Correct"]] if only_errors else out
    st.dataframe(view, width="stretch", height=380)
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---------------- Data preview ----------------
with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(raw_df.head(200), width="stretch")
    st.markdown("**Summary statistics (numeric features)**")
    st.dataframe(raw_df[meta["numeric_features"]].describe().T.round(3), width="stretch")
