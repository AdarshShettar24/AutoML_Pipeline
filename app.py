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
    "and get predictions easily."
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
    st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column Name", 0: "Data Type"}),
                 use_container_width=True, height=250)

    # ---------------------- AUTO CLEANING ----------------------
    if st.checkbox("🧠 Auto-clean: remove duplicates & fill missing values (recommended)"):
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            st.warning(f"🧾 Removed {removed} duplicate rows.")
        else:
            st.success("✅ No duplicate rows removed.")

        numeric_cols = detect_numeric_columns(df, min_non_na=1)
        df = safe_cast_numeric(df, numeric_cols)
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        cat_cols = [c for c in df.columns if c not in numeric_cols]
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                try:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                except Exception:
                    df[col] = df[col].fillna("")

        st.success("✨ Missing values filled (median for numeric, mode for categorical).")

        st.subheader("🧾 Cleaned Data Preview")
        st.dataframe(df.head(), use_container_width=True, height=250)
        with st.expander("🔍 View full cleaned dataset"):
            st.dataframe(df, use_container_width=True, height=400)

    # ---------------------- EDA ----------------------
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    num_cols = detect_numeric_columns(df, min_non_na=1)
    cat_cols = [c for c in df.columns if c not in num_cols]

    if st.checkbox("📈 Show Summary Statistics"):
        if len(num_cols) > 0:
            df_num = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            st.dataframe(df_num.describe().T, use_container_width=True, height=250)
        if len(cat_cols) > 0:
            cat_summary = pd.DataFrame(
                {col: [df[col].nunique(dropna=True),
                       df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else ""] for col in cat_cols},
                index=["Unique Values", "Most Frequent"]
            ).T
            st.dataframe(cat_summary)

    # ---------------------- HISTOGRAM SECTION ----------------------
    if st.checkbox("📊 Show Histograms"):
        # Sample only if dataset is huge
        plot_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 5000 else df
        cols_to_plot = st.multiselect("Choose columns to plot", num_cols, default=num_cols[:4])
        for col in cols_to_plot:
            series = pd.to_numeric(plot_df[col], errors="coerce").dropna()
            if not series.empty:
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.histplot(series, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
                plt.close(fig)

    # ---------------------- CORRELATION HEATMAP ----------------------
    if st.checkbox("📉 Show Correlation Heatmap"):
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns.")
        else:
            heatmap_df = df.sample(min(len(df), 3000), random_state=42) if len(df) > 5000 else df
            corr_df = heatmap_df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).corr()
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

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
