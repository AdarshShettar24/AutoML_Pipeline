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
st.title("🤖 Smart AutoML Dashboard (Explainable AI)")
st.markdown(
    "Upload a dataset, explore and clean it, train an AutoML model (PyCaret), "
    "and get simple plain-English explanations for models and predictions."
)

# ---------------------- SESSION STATE ----------------------
for key in ["trained_model", "is_classification", "target_column", "last_metrics", "train_columns"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📂 Upload Training CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    except UnicodeDecodeError:
        stringio = io.StringIO(uploaded_file.getvalue().decode("latin1"))

    try:
        df = pd.read_csv(stringio, index_col=False)
        df = df.astype(str)  # ✅ Fix: ensures consistent data types for Streamlit
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    if df.empty:
        st.warning("⚠️ The uploaded CSV file is empty.")
        st.stop()

    st.success("✅ Training data uploaded successfully!")

    # Show preview (first 5 rows)
    st.dataframe(df.head(), use_container_width=True, height=250)

    # Expandable full dataset
    with st.expander("🔍 View full dataset"):
        st.dataframe(df, use_container_width=True, height=400)

    st.write("📋 **Columns detected:**", list(df.columns))

    # ---------------------- DATA CLEANING SUMMARY ----------------------
    st.subheader("🧹 Data Cleaning Summary")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write("**Missing Values Detected:**")
        st.dataframe(missing.rename("Missing Count"))
    else:
        st.success("✅ No missing values detected!")

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"⚠️ Found {duplicates} duplicate rows.")
    else:
        st.success("✅ No duplicate rows found!")

    st.write("**Column Data Types:**")
    st.dataframe(
        df.dtypes.reset_index().rename(columns={"index": "Column Name", 0: "Data Type"}),
        use_container_width=True,
        height=250,
    )

    # ---------------------- AUTO CLEANING ----------------------
    if st.checkbox("🧠 Auto-clean: remove duplicates & fill missing values (recommended)"):
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            st.warning(f"🧾 Removed {removed} duplicate rows.")
        else:
            st.success("✅ No duplicate rows removed.")

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
        st.success("✨ Missing values filled (median for numeric, mode for categorical).")

        st.subheader("🧾 Cleaned Data Preview")
        st.dataframe(df.head(), use_container_width=True, height=250)
        with st.expander("🔍 View full cleaned dataset"):
            st.dataframe(df, use_container_width=True, height=400)

    # ---------------------- EDA ----------------------
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    if st.checkbox("📈 Show Summary Statistics"):
        st.markdown("### 🔢 Numeric Summary")
        if len(num_cols) > 0:
            st.dataframe(df[num_cols].describe().T, use_container_width=True, height=250)
        else:
            st.info("No numeric columns found.")

        st.markdown("### 🔤 Categorical Summary")
        if len(cat_cols) > 0:
            cat_summary = pd.DataFrame(
                {col: [df[col].nunique(), df[col].mode()[0]] for col in cat_cols},
                index=["Unique Values", "Most Frequent"],
            ).T
            st.dataframe(cat_summary)
        else:
            st.info("No categorical columns found.")

    # ---------------------- HISTOGRAM SECTION ----------------------
    if st.checkbox("Show Histograms (Numeric Columns)"):
        cols_to_plot = st.multiselect("Choose columns to plot", num_cols, default=num_cols[:4])
        for col in cols_to_plot:
            fig_large, ax_large = plt.subplots(figsize=(15, 5))
            sns.histplot(df[col].dropna().astype(float), kde=True, color="skyblue", ax=ax_large)
            ax_large.set_title(f"Distribution of {col}", fontsize=18)
            fig_large.tight_layout()
            st.pyplot(fig_large)
            plt.close(fig_large)

    # ---------------------- CORRELATION HEATMAP ----------------------
    if st.checkbox("Show Correlation Heatmap"):
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns for correlation heatmap.")
        else:
            corr_df = df[num_cols].astype(float).corr()
            fig, ax = plt.subplots(figsize=(15, 5))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=13)
            st.pyplot(fig)

    # ---------------------- TARGET SELECTION ----------------------
    if len(df.columns) > 1:
        target_column = st.selectbox("🎯 Select Target Column for Prediction", df.columns)
        st.session_state.target_column = target_column
    else:
        st.warning("⚠️ Not enough columns to choose a target variable.")
        st.stop()

    if df[target_column].dtype in ["int64", "float64"]:
        st.session_state.is_classification = False
        st.info("📈 Detected problem type: Regression (numeric target).")
    else:
        st.session_state.is_classification = True
        st.info("🧮 Detected problem type: Classification (categorical target).")

    # ---------------------- TRAINING ----------------------
    if st.button("🚀 Train Model"):
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])

        n_samples = len(df)
        n_folds = min(5, max(2, n_samples // 2))
        n_folds = min(n_folds, max(2, n_samples - 1))

        with st.spinner("⏳ Setting up and comparing models..."):
            if st.session_state.is_classification:
                counts = df[target_column].value_counts()
                df_filtered = df[df[target_column].isin(counts[counts >= 2].index)].reset_index(drop=True)
                if df_filtered.empty:
                    st.error("❌ Not enough samples per class (need at least 2 per class).")
                else:
                    cls_setup(data=df_filtered, target=target_column, verbose=False, index=False, session_id=42)
                    best_model = cls_compare(fold=n_folds, n_select=1)
                    leaderboard = cls_pull()
                    if 'Model' not in leaderboard.columns:
                        leaderboard.reset_index(inplace=True)
                        leaderboard.rename(columns={'index': 'Model'}, inplace=True)
                    st.subheader("🏆 All Model Leaderboard (with Model Names)")
                    st.dataframe(leaderboard, use_container_width=True)

                    try:
                        tuned = cls_tune(best_model, optimize="Accuracy", fold=n_folds)
                    except Exception:
                        tuned = best_model
                    st.session_state.trained_model = tuned
                    metrics = cls_pull()
                    st.session_state.last_metrics = metrics
                    st.session_state.train_columns = df_filtered.drop(columns=[target_column]).columns.tolist()
            else:
                reg_setup(data=df, target=target_column, verbose=False, index=False, session_id=42)
                best_model = reg_compare(fold=n_folds, n_select=1)
                leaderboard = reg_pull()
                if 'Model' not in leaderboard.columns:
                    leaderboard.reset_index(inplace=True)
                    leaderboard.rename(columns={'index': 'Model'}, inplace=True)
                st.subheader("🏆 All Model Leaderboard (with Model Names)")
                st.dataframe(leaderboard, use_container_width=True)

                try:
                    tuned = reg_tune(best_model, optimize="R2", fold=n_folds)
                except Exception:
                    tuned = best_model
                st.session_state.trained_model = tuned
                metrics = reg_pull()
                st.session_state.last_metrics = metrics
                st.session_state.train_columns = df.drop(columns=[target_column]).columns.tolist()

        st.success("✅ Model training finished.")

        st.subheader("🏁 Best Model Performance (Fold Results)")
        st.dataframe(st.session_state.last_metrics, use_container_width=True)

        try:
            model_name = type(st.session_state.trained_model).__name__
            st.markdown(f"### 🔎 Selected Best Model: **{model_name}**")
        except Exception:
            st.markdown("### 🔎 Selected Model: (information not available)")

        # Feature importance
        st.subheader("🌟 Feature Importance (if available)")
        try:
            fi = st.session_state.trained_model.feature_importances_
            feat_names = st.session_state.train_columns
            fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False)
            st.dataframe(fi_df.head(10))
        except Exception:
            st.info("Feature importance not available for this model.")

# ---------------------- PREDICTION SECTION ----------------------
st.subheader("🔮 Make Predictions on New Data")
new_file = st.file_uploader("📂 Upload New CSV for Prediction", type=["csv"], key="new_csv")

if new_file is not None:
    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        try:
            new_data = pd.read_csv(io.StringIO(new_file.getvalue().decode("utf-8")))
            new_data = new_data.astype(str)  # ✅ Fix: consistent data types
        except UnicodeDecodeError:
            new_data = pd.read_csv(io.StringIO(new_file.getvalue().decode("latin1")))
            new_data = new_data.astype(str)  # ✅ Fix

        st.write("📋 New Data Preview:")
        st.dataframe(new_data.head())

        with st.expander("🔍 View full new data"):
            st.dataframe(new_data, use_container_width=True, height=400)

        if st.button("✨ Predict"):
            with st.spinner("🔍 Generating predictions..."):
                if st.session_state.is_classification:
                    preds = cls_predict(st.session_state.trained_model, data=new_data)
                else:
                    preds = reg_predict(st.session_state.trained_model, data=new_data)

            st.subheader("🧾 Predictions")
            st.dataframe(preds)

            csv = preds.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
