import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Disease Prediction System", layout="wide")

st.title("Medical Data Disease Prediction System")
st.write("Compare **SVM, Logistic Regression, Random Forest, and XGBoost** across UCI medical datasets.")

@st.cache_resource
def load_assets():
    bundles = joblib.load("models/all_disease_models.pkl")
    metrics = pd.read_csv("models/metrics_comparison.csv")
    return bundles, metrics

try:
    bundles, metrics_df = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Please run `python download_data.py` and `python train_models.py` first.")
    st.stop()

# Sidebar Controls
st.sidebar.header("Configuration")
selected_disease = st.sidebar.selectbox("1. Select Disease Dataset", list(bundles.keys()))
disease_info = bundles[selected_disease]

selected_algo = st.sidebar.selectbox("2. Select Classification Algorithm", list(disease_info["models"].keys()))
active_model_info = disease_info["models"][selected_algo]
pipeline = active_model_info["pipeline"]

# Tabs for Performance Evaluation vs Live Prediction
tab1, tab2 = st.tabs(["Patient Risk Prediction", "Model Comparison & Confusion Matrix"])

with tab1:
    st.subheader(f"Predict {selected_disease} using {selected_algo}")
    st.caption("Adjust the patient clinical parameters below and click Predict.")

    features = disease_info["features"]
    stats = disease_info["feature_stats"]

    user_inputs = {}
    cols = st.columns(3)

    for i, feat in enumerate(features):
        col = cols[i % 3]
        f_min = stats[feat]["min"]
        f_max = stats[feat]["max"]
        f_mean = stats[feat]["mean"]
        step = 1.0 if (f_max - f_min) > 20 else 0.01

        user_inputs[feat] = col.number_input(
            label=feat,
            min_value=float(f_min * 0.5),
            max_value=float(f_max * 1.5 + 1.0),
            value=float(round(f_mean, 3)),
            step=step
        )

    if st.button("Run Diagnostic Prediction", type="primary"):
        input_df = pd.DataFrame([user_inputs])
        pred_class = pipeline.predict(input_df)[0]
        pred_prob = pipeline.predict_proba(input_df)[0][1]

        st.divider()
        if pred_class == 1:
            st.error(f"**Positive Detection ({selected_disease} Risk)** — Model Probability: **{pred_prob:.1%}**")
        else:
            st.success(f"**Negative Detection (Low {selected_disease} Risk)** — Disease Probability: **{pred_prob:.1%}**")

with tab2:
    st.subheader(f"Algorithm Performance on {selected_disease} Dataset")
    filtered_metrics = metrics_df[metrics_df["Disease"] == selected_disease].reset_index(drop=True)
    st.dataframe(filtered_metrics, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Confusion Matrix ({selected_algo})**")
        cm = np.array(active_model_info["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Healthy (0)", "Disease (1)"],
                    yticklabels=["Healthy (0)", "Disease (1)"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_b:
        st.markdown("**ROC-AUC & Recall Comparison**")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        x = np.arange(len(filtered_metrics["Algorithm"]))
        width = 0.35
        ax2.bar(x - width/2, filtered_metrics["Recall"], width, label="Recall")
        ax2.bar(x + width/2, filtered_metrics["ROC-AUC"], width, label="ROC-AUC")
        ax2.set_xticks(x)
        ax2.set_xticklabels(filtered_metrics["Algorithm"], rotation=15)
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        st.pyplot(fig2)