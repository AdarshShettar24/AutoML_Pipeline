# app.py
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

# ---------------------- HELPER FUNCTIONS ----------------------
def detect_numeric_columns(df):
    """Detect numeric columns safely."""
    return df.select_dtypes(include=["int", "float"]).columns.tolist()

def safe_cast_numeric(df, numeric_cols):
    """Convert numeric columns safely to float."""
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="AutoML Pipeline", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Pipeline for Classification & Regression")

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.write("### 🧾 Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("### 📊 Summary Statistics")
    st.write("📘 This shows mean, standard deviation, min, max, and quartiles for numeric columns.")
    st.dataframe(df.describe())

    # ---------------------- VISUALIZATIONS ----------------------
    st.subheader("📈 Visualizations")
    numeric_cols = detect_numeric_columns(df)

    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        df[numeric_cols].hist(bins=20, figsize=(12, 6))
        st.pyplot(plt.gcf())

        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns detected for visualization.")

    # ---------------------- TARGET SELECTION ----------------------
    st.subheader("🎯 Target Column Selection")
    target_column = st.selectbox("Select your target column", df.columns)

    problem_type = st.radio("Select the type of problem:", ("Classification", "Regression"))

    if problem_type == "Classification":
        st.session_state.is_classification = True
    else:
        st.session_state.is_classification = False

    # ---------------------- TRAINING ----------------------
    if st.button("🚀 Train Model"):
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])

        numeric_cols = detect_numeric_columns(df)
        df = safe_cast_numeric(df, numeric_cols)
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(float)

        n_samples = len(df)
        if n_samples > 8000:
            st.warning("⚡ Large dataset detected — sampling 3000 rows for faster training.")
            df = df.sample(3000, random_state=42)
            n_folds = 2
        elif n_samples > 3000:
            n_folds = 3
            st.info("⚡ Medium dataset — reduced folds to speed up training.")
        else:
            n_folds = 5

        with st.spinner("⏳ Setting up and comparing models..."):
            if st.session_state.is_classification:
                cls_setup(
                    data=df,
                    target=target_column,
                    verbose=False,
                    html=False,
                    silent=True,
                    session_id=42,
                    fold_strategy="kfold",
                    fold=n_folds,
                    normalize=True,
                    transformation=False,
                    feature_selection=False
                )
                best_model = cls_compare(fold=n_folds, turbo=True)
                leaderboard = cls_pull()
            else:
                reg_setup(
                    data=df,
                    target=target_column,
                    verbose=False,
                    html=False,
                    silent=True,
                    session_id=42,
                    fold_strategy="kfold",
                    fold=n_folds,
                    normalize=True,
                    transformation=False,
                    feature_selection=False
                )
                best_model = reg_compare(fold=n_folds, turbo=True)
                leaderboard = reg_pull()

        st.session_state.trained_model = best_model
        st.session_state.last_metrics = leaderboard
        st.session_state.train_columns = df.drop(columns=[target_column]).columns.tolist()
        st.success("✅ Model training completed!")

        st.subheader("🏁 Model Leaderboard")
        st.dataframe(leaderboard, use_container_width=True)

        try:
            model_name = type(best_model).__name__
            st.markdown(f"### 🔎 Selected Best Model: **{model_name}**")
        except Exception:
            st.markdown("### 🔎 Model Info Not Available")

        # Feature importance
        st.subheader("🌟 Feature Importance (if available)")
        try:
            fi = best_model.feature_importances_
            feat_names = st.session_state.train_columns
            fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi}).sort_values("Importance", ascending=False)
            st.dataframe(fi_df.head(10))
        except Exception:
            st.info("Feature importance not available for this model.")

    # ---------------------- PREDICTION ----------------------
    st.subheader("🔮 Make Predictions")
    predict_file = st.file_uploader("Upload new dataset for prediction (same columns as training)", type=["csv"])

    if predict_file and "trained_model" in st.session_state:
        new_data = pd.read_csv(predict_file)
        st.write("📄 Prediction Data Preview")
        st.dataframe(new_data.head())

        if st.session_state.is_classification:
            preds = cls_predict(st.session_state.trained_model, new_data)
        else:
            preds = reg_predict(st.session_state.trained_model, new_data)

        st.subheader("🧾 Predictions")
        st.dataframe(preds.head())

        csv = preds.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("👈 Upload a CSV file to get started.")
