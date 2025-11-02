import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import warnings
import joblib

from pycaret.classification import (
    setup as cls_setup,
    compare_models as cls_compare,
    predict_model as cls_predict,
    pull as cls_pull,
    tune_model as cls_tune,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    predict_model as reg_predict,
    pull as reg_pull,
    tune_model as reg_tune,
)

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="🤖 Smart AutoML Dashboard")

# ---------------------- APP TITLE ----------------------
st.title("🤖 Smart AutoML Dashboard")
st.markdown(
    "Upload a dataset, explore and clean it, train an AutoML model (PyCaret), "
    "and get predictions with ease."
)

# ---------------------- SESSION STATE ----------------------
for key in ["trained_model", "is_classification", "target_column", "last_metrics", "train_columns"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ---------- Helper utilities ----------
def detect_numeric_columns(df, min_non_na=1):
    numeric_cols = []
    for c in df.columns:
        non_na = pd.to_numeric(df[c], errors="coerce").notna().sum()
        if non_na >= min_non_na:
            numeric_cols.append(c)
    return numeric_cols


def safe_cast_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def detect_target_type(df, target_col):
    num_nonnull = pd.to_numeric(df[target_col], errors="coerce").notna().sum()
    prop_numeric = num_nonnull / max(1, len(df))
    return False if prop_numeric >= 0.9 else True


# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📂 Upload Training CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    except UnicodeDecodeError:
        stringio = io.StringIO(uploaded_file.getvalue().decode("latin1"))

    try:
        df = pd.read_csv(stringio, index_col=False)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    if df.empty:
        st.warning("⚠️ The uploaded CSV file is empty.")
        st.stop()

    df = df.dropna(how="all").reset_index(drop=True)
    st.success("✅ Training data uploaded successfully!")

    st.dataframe(df.head(), use_container_width=True, height=250)
    with st.expander("🔍 View full dataset"):
        st.dataframe(df, use_container_width=True, height=400)

    st.write("📋 **Columns detected:**", list(df.columns))

    # ---------------------- TARGET SELECTION ----------------------
    if len(df.columns) > 1:
        target_column = st.selectbox("🎯 Select Target Column for Prediction", df.columns)
        st.session_state.target_column = target_column
    else:
        st.warning("⚠️ Not enough columns to choose a target variable.")
        st.stop()

    is_class = detect_target_type(df, target_column)
    st.session_state.is_classification = is_class
    st.info("📈 Problem Type: " + ("Classification" if is_class else "Regression"))

    # ---------------------- TRAINING ----------------------
    if st.button("🚀 Train Model"):
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])

        numeric_cols = detect_numeric_columns(df)
        df = safe_cast_numeric(df, numeric_cols)
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(float)

        n_samples = len(df)

        # ⚡ Smart speed modes
        if n_samples <= 2000:
            fast_mode = True
            n_folds = 3
            turbo_mode = True
            st.info("⚡ Small dataset detected — Fast Mode enabled (lightweight models only).")
        elif n_samples <= 5000:
            fast_mode = False
            n_folds = 3
            turbo_mode = True
            st.info("⚡ Medium dataset detected — using 3-fold CV (turbo mode).")
        else:
            fast_mode = False
            n_folds = 2
            turbo_mode = True
            st.warning("⚡ Large dataset detected — sampling 3000 rows for faster training.")
            df = df.sample(3000, random_state=42)

        with st.spinner("⏳ Setting up and training models..."):
            if st.session_state.is_classification:
                cls_setup(
                    data=df,
                    target=target_column,
                    verbose=False,
                    index=False,
                    session_id=42,
                    fold=n_folds,
                    fix_imbalance=False,
                    feature_selection=False,
                    profile=False,
                )

                if fast_mode:
                    from pycaret.classification import create_model
                    models_to_try = ["lr", "dt", "knn", "nb"]
                    best_model = max(
                        [create_model(m) for m in models_to_try],
                        key=lambda m: cls_pull()["Accuracy"].iloc[-1],
                    )
                else:
                    best_model = cls_compare(fold=n_folds, turbo=turbo_mode)

                leaderboard = cls_pull()
                st.subheader("🏆 Model Leaderboard")
                st.dataframe(leaderboard, use_container_width=True)

                st.session_state.trained_model = best_model
                st.session_state.last_metrics = leaderboard
                st.session_state.train_columns = df.drop(columns=[target_column]).columns.tolist()

            else:
                reg_setup(
                    data=df,
                    target=target_column,
                    verbose=False,
                    index=False,
                    session_id=42,
                    fold=n_folds,
                    feature_selection=False,
                    profile=False,
                )

                if fast_mode:
                    from pycaret.regression import create_model
                    models_to_try = ["lr", "dt", "lasso", "ridge"]
                    best_model = max(
                        [create_model(m) for m in models_to_try],
                        key=lambda m: reg_pull()["R2"].iloc[-1],
                    )
                else:
                    best_model = reg_compare(fold=n_folds, turbo=turbo_mode)

                leaderboard = reg_pull()
                st.subheader("🏆 Model Leaderboard")
                st.dataframe(leaderboard, use_container_width=True)

                st.session_state.trained_model = best_model
                st.session_state.last_metrics = leaderboard
                st.session_state.train_columns = df.drop(columns=[target_column]).columns.tolist()

        st.success("✅ Model training completed!")

        st.subheader("🏁 Best Model Performance")
        st.dataframe(st.session_state.last_metrics, use_container_width=True)

        try:
            model_name = type(st.session_state.trained_model).__name__
            st.markdown(f"### 🔎 Selected Best Model: **{model_name}**")
        except Exception:
            st.markdown("### 🔎 Selected Model: (not available)")

# ---------------------- PREDICTION SECTION ----------------------
st.subheader("🔮 Make Predictions on New Data")
new_file = st.file_uploader("📂 Upload New CSV for Prediction", type=["csv"], key="new_csv")

if new_file is not None:
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        try:
            new_data = pd.read_csv(io.StringIO(new_file.getvalue().decode("utf-8")))
        except UnicodeDecodeError:
            new_data = pd.read_csv(io.StringIO(new_file.getvalue().decode("latin1")))

        train_cols = st.session_state.get("train_columns")
        if train_cols:
            for c in train_cols:
                if c not in new_data.columns:
                    new_data[c] = pd.NA
            new_data = new_data.reindex(columns=train_cols)

        st.write("📋 New Data Preview:")
        st.dataframe(new_data.head())

        if st.button("✨ Predict"):
            with st.spinner("🔍 Generating predictions..."):
                if st.session_state.is_classification:
                    preds = cls_predict(st.session_state.trained_model, data=new_data)
                else:
                    preds = reg_predict(st.session_state.trained_model, data=new_data)

            st.subheader("🧾 Predictions")
            st.dataframe(preds)
            csv = preds.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
