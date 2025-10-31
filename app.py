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
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    predict_model as reg_predict,
    pull as reg_pull,
)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🤖 Smart AutoML Dashboard (Explainable AI)", layout="wide")

st.title("🤖 Smart AutoML Dashboard (Explainable AI)")
st.markdown("### Upload your dataset to begin")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    
    # Show limited preview to avoid "View smaller version" message
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(100))

    # Dataset overview
    st.subheader("🧭 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Columns Detected:**", list(df.columns))

    # Data Cleaning Summary
    st.subheader("🧹 Data Cleaning Summary")
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    if missing == 0:
        st.info("✅ No missing values detected!")
    else:
        st.warning(f"⚠️ {missing} missing values detected.")

    if duplicates == 0:
        st.info("✅ No duplicate rows found!")
    else:
        st.warning(f"⚠️ {duplicates} duplicate rows found.")

    st.write("**Column Data Types:**")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Data Type"}))

    # Auto Clean
    if st.button("✨ Auto Clean Data"):
        df_clean = df.copy()

        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        # Remove duplicates
        before = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        after = len(df_clean)
        removed = before - after

        st.success(f"🧼 No duplicate rows removed." if removed == 0 else f"🧼 {removed} duplicate rows removed.")
        st.info("✨ Missing values filled (median for numeric, mode for categorical).")

        st.subheader("🧾 Cleaned Data Preview")
        st.dataframe(df_clean.head(100))

        st.session_state.df_clean = df_clean

    # Perform EDA
    if "df_clean" in st.session_state:
        df_clean = st.session_state.df_clean
        st.subheader("📈 Exploratory Data Analysis (EDA)")

        if st.checkbox("Show Summary Statistics"):
            st.markdown("📘 This shows mean, standard deviation, min, max, and quartiles for numeric columns.")
            st.dataframe(df_clean.describe().T)

        if st.checkbox("Show Histograms"):
            numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df_clean[col], kde=True, ax=ax)
                st.pyplot(fig)

        if st.checkbox("Show Correlation Heatmap"):
            corr = df_clean.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Select Target Column
        st.subheader("🎯 Select Target Column for Prediction")
        target = st.selectbox("Choose Target Column", options=df_clean.columns)

        if st.button("🚀 Train Model"):
            y = df_clean[target]
            X = df_clean.drop(columns=[target])

            # Detect problem type
            problem_type = "Classification" if y.nunique() <= 20 and y.dtypes != "float64" else "Regression"
            st.info(f"Detected Problem Type: **{problem_type}**")

            if problem_type == "Classification":
                cls = cls_setup(df_clean, target=target, silent=True, session_id=123)
                best_model = cls_compare()
                st.session_state.trained_model = best_model
                st.subheader("🏆 All Model Leaderboard")
                st.dataframe(cls_pull())

                st.subheader("🔥 Best Model Performance (Fold Results)")
                st.dataframe(cls_pull())

                st.subheader("✅ Selected Best Model")
                st.write(best_model)

                try:
                    st.subheader("📊 Feature Importance")
                    fig = cls_pull()
                    st.pyplot(fig)
                except:
                    st.info("No feature importance available for this model.")

            else:
                reg = reg_setup(df_clean, target=target, silent=True, session_id=123)
                best_model = reg_compare()
                st.session_state.trained_model = best_model
                st.subheader("🏆 All Model Leaderboard")
                st.dataframe(reg_pull())

                st.subheader("🔥 Best Model Performance (Fold Results)")
                st.dataframe(reg_pull())

                st.subheader("✅ Selected Best Model")
                st.write(best_model)

                try:
                    st.subheader("📊 Feature Importance")
                    fig = reg_pull()
                    st.pyplot(fig)
                except:
                    st.info("No feature importance available for this model.")

    # Prediction on new data
    if "trained_model" in st.session_state:
        st.subheader("📤 Make Predictions on New Data")
        new_file = st.file_uploader("Upload new dataset for prediction", type=["csv"], key="newdata")

        if new_file:
            new_data = pd.read_csv(new_file)
            st.write("📄 New Data Preview:")
            st.dataframe(new_data.head(100))

            if st.button("🔮 Predict"):
                try:
                    if "Classification" in str(type(st.session_state.trained_model)):
                        preds = cls_predict(st.session_state.trained_model, data=new_data)
                    else:
                        preds = reg_predict(st.session_state.trained_model, data=new_data)

                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(preds.head(100))

                    # Download predictions
                    csv = preds.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error("❌ Prediction failed. Please make sure columns match training data.")
                    st.exception(e)
