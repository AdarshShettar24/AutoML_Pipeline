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


# ---------- Helper utilities ----------
def detect_numeric_columns(df, min_non_na=1):
    """Return list of columns that contain at least min_non_na numeric-parsable values."""
    numeric_cols = []
    for c in df.columns:
        non_na = pd.to_numeric(df[c], errors="coerce").notna().sum()
        if non_na >= min_non_na:
            numeric_cols.append(c)
    return numeric_cols


def safe_cast_numeric(df, cols):
    """Cast listed cols to float (in-place) using to_numeric coercing errors to NaN."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def detect_target_type(df, target_col):
    """Return True if classification (categorical), False if regression (numeric)"""
    # Proportion of numeric-parsable values
    num_nonnull = pd.to_numeric(df[target_col], errors="coerce").notna().sum()
    prop_numeric = num_nonnull / max(1, len(df))
    # If majority numeric -> regression
    if prop_numeric >= 0.9:
        return False  # regression
    else:
        return True  # classification


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

    # Drop fully empty rows (if any)
    df = df.dropna(how="all").reset_index(drop=True)

    st.success("✅ Training data uploaded successfully!")

    # Show preview (first 5 rows)
    st.dataframe(df.head(), use_container_width=True, height=250)

    # Expandable section to show full dataset
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
    # Show pandas dtypes (original)
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

        # detect numeric columns robustly
        numeric_cols = detect_numeric_columns(df, min_non_na=1)
        # cast numeric cols
        df = safe_cast_numeric(df, numeric_cols)

        # fill numeric
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # categorical columns = those not in numeric_cols
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

    # detect numeric columns for EDA (require at least 1 numeric-parsable value)
    num_cols = detect_numeric_columns(df, min_non_na=1)
    cat_cols = [c for c in df.columns if c not in num_cols]

    if st.checkbox("📈 Show Summary Statistics"):
        st.markdown("### 🔢 Numeric Summary")
        if len(num_cols) > 0:
            # show describe for numeric columns (coerced to numeric)
            df_num = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            st.dataframe(df_num.describe().T, use_container_width=True, height=250)
            st.markdown(
                """
                <div style='font-size:30px; color:#E65100; font-weight:500; margin-top:10px;'>
                📘 This shows <b>mean, standard deviation, min, max,</b> and <b>quartiles</b> for numeric columns.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No numeric columns found.")

        st.markdown("### 🔤 Categorical Summary")
        if len(cat_cols) > 0:
            cat_summary = pd.DataFrame(
                {col: [df[col].nunique(dropna=True), df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else ""] for col in cat_cols},
                index=["Unique Values", "Most Frequent"],
            ).T
            st.dataframe(cat_summary)
            st.markdown(
                """
                <div style='font-size:30px; color:#E65100; font-weight:500; margin-top:10px;'>
                📗 Shows <b>number of unique values</b> and the <b>most frequent category</b> for each categorical column.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No categorical columns found.")

    # ---------------------- HISTOGRAM SECTION ----------------------
    if st.checkbox("Show Histograms (Numeric Columns)"):
        cols_to_plot = st.multiselect("Choose columns to plot", num_cols, default=num_cols[:4])
        for col in cols_to_plot:
            # coerce to numeric for plotting
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                st.info(f"Column {col} has no numeric values to plot.")
                continue
            fig_large, ax_large = plt.subplots(figsize=(15, 5))
            sns.histplot(series, kde=True, ax=ax_large)
            ax_large.set_title(f"Distribution of {col}", fontsize=18)
            fig_large.tight_layout()
            st.pyplot(fig_large)
            plt.close(fig_large)

            st.markdown(
                f"""
                <div style='font-size:30px; color:#E65100; font-weight:500; margin-top:10px;'>
                🧠 <b>Interpretation:</b> Histogram for <b>{col}</b> — peaks show where most values lie.<br>
                Skew left/right indicates bias; narrow peak = consistent values; wide spread = high variability.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------- CORRELATION HEATMAP ----------------------
    if st.checkbox("Show Correlation Heatmap"):
        if len(num_cols) < 2:
            st.info("Need at least two numeric columns for correlation heatmap.")
        else:
            corr_df = df[num_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).corr()
            fig, ax = plt.subplots(figsize=(15, 5))
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=13)
            st.pyplot(fig)
            st.markdown(
                """
                <div style='font-size:30px; color:#E65100; font-weight:500; margin-top:5px;'>
                🧠 <b>Interpretation:</b> Heatmap shows relationships between numeric columns.<br>
                +1 = strong positive, -1 = strong negative, 0 = no linear relation.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------- TARGET SELECTION ----------------------
    if len(df.columns) > 1:
        target_column = st.selectbox("🎯 Select Target Column for Prediction", df.columns)
        st.session_state.target_column = target_column
    else:
        st.warning("⚠️ Not enough columns to choose a target variable.")
        st.stop()

    # Determine task type robustly
    is_class = detect_target_type(df, target_column)
    st.session_state.is_classification = is_class
    if not is_class:
        st.info("📈 Detected problem type: Regression (numeric target).")
    else:
        st.info("🧮 Detected problem type: Classification (categorical target).")

    # ---------------------- TRAINING ----------------------
    if st.button("🚀 Train Model"):
        if "Name" in df.columns:
            df = df.drop(columns=["Name"])

        # Ensure numeric columns are properly typed for PyCaret
        numeric_cols = detect_numeric_columns(df, min_non_na=1)
        df = safe_cast_numeric(df, numeric_cols)
        # Cast numeric columns to float explicitly (PyCaret friendly)
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].astype(float)

        # Adaptive folds (keep original behaviour for small sets)
        n_samples = len(df)
        n_folds = min(5, max(2, n_samples // 2))
        n_folds = min(n_folds, max(2, n_samples - 1))

        # Speed safeguard: sample if extremely large
        if n_samples > 5000:
            st.warning("⚠️ Very large dataset detected — sampling 3000 rows to avoid timeouts.")
            df = df.sample(3000, random_state=42)
            n_folds = 3

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
            fi_df = pd.DataFrame(
                {"feature": feat_names, "importance": fi}
            ).sort_values("importance", ascending=False)
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

        # Align new_data columns to training columns if available
        train_cols = st.session_state.get("train_columns")
        if train_cols:
            # Add missing training columns to new_data with NaN
            for c in train_cols:
                if c not in new_data.columns:
                    new_data[c] = pd.NA
            # Keep only train columns in same order
            new_data = new_data.reindex(columns=train_cols)

        st.write("📋 New Data Preview:")
        st.dataframe(new_data.head())

        with st.expander("🔍 View full new data"):
            st.dataframe(new_data, use_container_width=True, height=400)

        if st.button("✨ Predict"):
            with st.spinner("🔍 Generating predictions..."):
                # Make sure numeric columns in new_data are numeric when needed
                if st.session_state.train_columns:
                    numeric_cols = detect_numeric_columns(pd.concat([pd.DataFrame([], columns=st.session_state.train_columns), new_data], ignore_index=True), min_non_na=1)
                    new_data = safe_cast_numeric(new_data, numeric_cols)
                    if len(numeric_cols) > 0:
                        new_data[numeric_cols] = new_data[numeric_cols].astype(float)

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
                mime="text/csv",
            )
