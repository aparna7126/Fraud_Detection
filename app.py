
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# LOAD SAVED MODEL ARTIFACTS AND DASHBOARD DATA
# --------------------------------------------------

@st.cache_resource
def load_model_artifacts():
    with open(os.path.join(MODEL_DIR, "fraudshield_random_forest.pkl"), "rb") as f:
        model = joblib.load(f)

    with open(os.path.join(MODEL_DIR, "fraudshield_preprocessor.pkl"), "rb") as f:
        preprocessor = joblib.load(f)

    with open(os.path.join(MODEL_DIR, "selected_feature_indices.pkl"), "rb") as f:
        selected_indices = joblib.load(f)

    with open(os.path.join(MODEL_DIR, "all_model_features.pkl"), "rb") as f:
        features = joblib.load(f)

    with open(os.path.join(MODEL_DIR, "fraud_alert_threshold.pkl"), "rb") as f:
        threshold = joblib.load(f)

    return model, preprocessor, selected_indices, features, threshold


@st.cache_data
def load_dashboard_data():
    dashboard = pd.read_csv(
        os.path.join(MODEL_DIR, "fraudshield_dashboard_data.csv")
    )
    daily = pd.read_csv(
        os.path.join(MODEL_DIR, "fraudshield_daily_hotspots.csv")
    )
    hourly = pd.read_csv(
        os.path.join(MODEL_DIR, "fraudshield_hourly_hotspots.csv")
    )
    return dashboard, daily, hourly


model, preprocessor, selected_indices, features, threshold = load_model_artifacts()
dashboard, daily, hourly = load_dashboard_data()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ FraudShield AI")
st.sidebar.caption("Intelligent Transaction Fraud Detection")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview Dashboard",
        "Real-Time Fraud Alerts",
        "Fraud Hotspot Calendar",
        "Transaction Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.metric("Detection Threshold", f"{threshold:.4f}")
st.sidebar.caption("Test dataset contains unlabeled historical transactions.")

# --------------------------------------------------
# OVERVIEW DASHBOARD
# --------------------------------------------------

if page == "Overview Dashboard":

    st.title("🛡️ FraudShield AI Dashboard")
    st.write("Machine Learning Based Financial Fraud Monitoring")

    total = len(dashboard)
    alerts = int(dashboard["Predicted_Fraud"].sum())
    genuine = total - alerts
    alert_rate = alerts / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Potential Fraud Alerts", f"{alerts:,}")
    c3.metric("Predicted Genuine", f"{genuine:,}")
    c4.metric("Alert Rate", f"{alert_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Risk Distribution")
        risk_counts = (
            dashboard["Risk_Level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .reset_index()
        )
        risk_counts.columns = ["Risk Level", "Transactions"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Transactions",
            hole=0.55,
            title="Predicted Risk Categories"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Predicted Fraud Alerts by Risk")
        alert_counts = (
            dashboard[dashboard["Predicted_Fraud"] == 1]
            ["Risk_Level"]
            .value_counts()
            .reindex(["Medium", "High"])
            .fillna(0)
            .reset_index()
        )
        alert_counts.columns = ["Risk Level", "Alerts"]

        fig = px.bar(
            alert_counts,
            x="Risk Level",
            y="Alerts",
            text="Alerts",
            title="Potential Fraud Alerts"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Recent High-Risk Transactions")
    high_risk = dashboard[
        dashboard["Predicted_Fraud"] == 1
    ].sort_values(
        "Fraud_Probability",
        ascending=False
    ).head(10)

    st.dataframe(high_risk, width="stretch")

# --------------------------------------------------
# REAL-TIME FRAUD ALERT SIMULATION
# --------------------------------------------------

elif page == "Real-Time Fraud Alerts":

    st.title("🚨 Real-Time Fraud Alert System")
    st.warning(
        "Demonstration mode: transactions are replayed from the historical "
        "unlabeled test dataset. This is not a live banking feed."
    )

    st.subheader("Simulate Incoming Transactions")

    max_index = len(dashboard) - 1

    start_index = st.number_input(
        "Starting transaction row",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1
    )

    batch_size = st.slider(
        "Number of transactions to simulate",
        min_value=1,
        max_value=100,
        value=10
    )

    if st.button("Process Transactions", type="primary"):

        batch = dashboard.iloc[
            int(start_index):int(start_index) + batch_size
        ].copy()

        for _, transaction in batch.iterrows():

            probability = transaction["Fraud_Probability"]
            is_alert = transaction["Predicted_Fraud"] == 1

            if probability >= 0.90:
                st.error(
                    f"HIGH RISK ALERT | Transaction "
                    f"{int(transaction['TransactionID'])} | "
                    f"Probability: {probability:.2%}"
                )
            elif is_alert:
                st.warning(
                    f"MEDIUM RISK ALERT | Transaction "
                    f"{int(transaction['TransactionID'])} | "
                    f"Probability: {probability:.2%}"
                )
            else:
                st.success(
                    f"LOW RISK | Transaction "
                    f"{int(transaction['TransactionID'])} | "
                    f"Probability: {probability:.2%}"
                )

        st.subheader("Processed Transaction Details")
        st.dataframe(batch, width="stretch")

# --------------------------------------------------
# FRAUD HOTSPOT CALENDAR
# --------------------------------------------------

elif page == "Fraud Hotspot Calendar":

    st.title("📅 Fraud Hotspot Calendar")
    st.write(
        "Explore relative transaction days and hours with elevated "
        "model-predicted fraud activity."
    )

    st.info(
        "Relative day numbers represent elapsed time bins, not real calendar dates. "
        "Hotspots show model predictions, not confirmed fraud."
    )

    st.subheader("Daily Predicted Fraud Alerts")

    fig = px.bar(
        daily,
        x="TransactionDay",
        y="Predicted_Alerts",
        color="Alert_Percentage",
        color_continuous_scale="Reds",
        title="Predicted Alerts by Relative Day",
        hover_data=["Total_Transactions", "Average_Risk", "Alert_Percentage"]
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Hourly Predicted Fraud Activity")

    hourly_display = hourly.copy()
    hourly_display["Hour_Label"] = (
        hourly_display["TransactionHour"].astype(str).str.zfill(2) + ":00"
    )

    fig = px.bar(
        hourly_display,
        x="Hour_Label",
        y="Alert_Percentage",
        color="Alert_Percentage",
        color_continuous_scale="Reds",
        text="Alert_Percentage",
        title="Predicted Alert Percentage by Hour"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 Relative-Day Hotspots")
    st.dataframe(
        daily.nlargest(10, "Predicted_Alerts"),
        width="stretch"
    )

# --------------------------------------------------
# TRANSACTION EXPLORER
# --------------------------------------------------

elif page == "Transaction Explorer":

    st.title("🔎 Transaction Explorer")

    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )

    minimum_probability = st.slider(
        "Minimum Fraud Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

    filtered = dashboard[
        dashboard["Risk_Level"].isin(risk_filter)
        & (dashboard["Fraud_Probability"] >= minimum_probability)
    ]

    st.metric("Matching Transactions", f"{len(filtered):,}")

    st.dataframe(
        filtered.head(1000),
        width="stretch"
    )

    st.download_button(
        "Download Filtered Transactions",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="fraudshield_filtered_transactions.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("FraudShield AI | CSE303 Capstone Project")
st.sidebar.caption("Machine Learning Fraud Detection")
