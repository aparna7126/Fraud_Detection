import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
# FILE PATH HANDLING
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(MODEL_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def find_file(filename, search_dirs):
    """Find a file in the specified directories."""

    for directory in search_dirs:
        path = os.path.join(directory, filename)

        if os.path.exists(path):
            return path

    return None


SEARCH_DIRS = [MODEL_DIR, RESULTS_DIR, PROJECT_DIR]


# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_model_artifacts():

    required_files = [
        "fraudshield_random_forest.pkl",
        "fraudshield_preprocessor.pkl",
        "selected_feature_indices.pkl",
        "all_model_features.pkl",
        "fraud_alert_threshold.pkl"
    ]

    artifact_paths = {}

    for filename in required_files:

        path = find_file(filename, SEARCH_DIRS)

        if path is None:
            raise FileNotFoundError(
                f"Required model file not found: {filename}"
            )

        artifact_paths[filename] = path

    model = joblib.load(
        artifact_paths["fraudshield_random_forest.pkl"]
    )

    preprocessor = joblib.load(
        artifact_paths["fraudshield_preprocessor.pkl"]
    )

    selected_indices = joblib.load(
        artifact_paths["selected_feature_indices.pkl"]
    )

    features = joblib.load(
        artifact_paths["all_model_features.pkl"]
    )

    threshold = joblib.load(
        artifact_paths["fraud_alert_threshold.pkl"]
    )

    return model, preprocessor, selected_indices, features, threshold


# --------------------------------------------------
# LOAD DASHBOARD DATA SAFELY
# --------------------------------------------------

@st.cache_data
def load_dashboard_data():

    filenames = [
        "fraudshield_dashboard_data.csv",
        "fraudshield_daily_hotspots.csv",
        "fraudshield_hourly_hotspots.csv"
    ]

    dataframes = []

    for filename in filenames:

        path = find_file(filename, SEARCH_DIRS)

        if path is None:
            raise FileNotFoundError(
                f"Dashboard file not found: {filename}. "
                f"Please upload it to the app folder or results folder."
            )

        dataframes.append(pd.read_csv(path))

    return tuple(dataframes)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    model, preprocessor, selected_indices, features, threshold = (
        load_model_artifacts()
    )

except Exception as e:

    st.error("Unable to load the trained model artifacts.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ FraudShield AI")
st.sidebar.caption("Intelligent Transaction Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview Dashboard",
        "Transaction Prediction",
        "Real-Time Fraud Alerts",
        "Fraud Hotspot Calendar",
        "Transaction Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.metric(
    "Detection Threshold",
    f"{threshold:.4f}"
)

st.sidebar.caption(
    "Test dataset contains unlabeled historical transactions."
)


# --------------------------------------------------
# OVERVIEW DASHBOARD
# --------------------------------------------------

if page == "Overview Dashboard":

    st.title("🛡️ FraudShield AI Dashboard")

    st.write(
        "Machine Learning Based Financial Fraud Monitoring"
    )

    try:

        dashboard, daily, hourly = load_dashboard_data()

    except Exception as e:

        st.error("Unable to load dashboard CSV files.")
        st.exception(e)
        st.stop()

    total = len(dashboard)

    alerts = int(
        dashboard["Predicted_Fraud"].sum()
    )

    genuine = total - alerts

    alert_rate = alerts / total * 100 if total else 0

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

        risk_counts.columns = [
            "Risk Level",
            "Transactions"
        ]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Transactions",
            hole=0.55,
            title="Predicted Risk Categories"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Predicted Fraud Alerts by Risk")

        alert_counts = (
            dashboard[
                dashboard["Predicted_Fraud"] == 1
            ]["Risk_Level"]
            .value_counts()
            .reindex(["Medium", "High"])
            .fillna(0)
            .reset_index()
        )

        alert_counts.columns = [
            "Risk Level",
            "Alerts"
        ]

        fig = px.bar(
            alert_counts,
            x="Risk Level",
            y="Alerts",
            text="Alerts",
            title="Potential Fraud Alerts"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Recent High-Risk Transactions")

    high_risk = dashboard[
        dashboard["Predicted_Fraud"] == 1
    ].sort_values(
        "Fraud_Probability",
        ascending=False
    ).head(10)

    st.dataframe(
        high_risk,
        use_container_width=True
    )


# --------------------------------------------------
# TRANSACTION PREDICTION USING USER UPLOAD
# --------------------------------------------------

elif page == "Transaction Prediction":

    st.title("🔍 Transaction Fraud Prediction")

    st.write(
        "Upload transaction data in CSV format to evaluate "
        "transaction risk using the trained Random Forest model."
    )

    st.info(
        "The uploaded data must contain the original input features "
        "used by the trained preprocessing pipeline. "
        "Do not include the fraud label column."
    )

    st.subheader("Upload Transaction CSV")

    uploaded_file = st.file_uploader(
        "Upload your transaction CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            input_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")

            st.dataframe(
                input_data.head(10),
                use_container_width=True
            )

            st.write(
                f"Uploaded transactions: **{len(input_data):,}**"
            )

            # --------------------------------------------------
            # CHECK REQUIRED FEATURES
            # --------------------------------------------------

            required_features = list(features)

            missing_features = [
                feature
                for feature in required_features
                if feature not in input_data.columns
            ]

            extra_features = [
                column
                for column in input_data.columns
                if column not in required_features
            ]

            if missing_features:

                st.error(
                    "The uploaded CSV is missing "
                    f"{len(missing_features)} required features."
                )

                st.write("Missing columns:")

                st.code(
                    ", ".join(
                        map(str, missing_features)
                    )
                )

            else:

                if extra_features:

                    st.warning(
                        "Extra columns will be ignored: "
                        + ", ".join(
                            map(str, extra_features[:20])
                        )
                    )

                # Arrange columns in the original model order

                input_data = input_data[
                    required_features
                ].copy()

                # --------------------------------------------------
                # PREDICT
                # --------------------------------------------------

                if st.button(
                    "Predict Transaction Risk",
                    type="primary"
                ):

                    with st.spinner(
                        "Analyzing transactions..."
                    ):

                        transformed = (
                            preprocessor.transform(input_data)
                        )

                        selected_data = transformed[
                            :, selected_indices
                        ]

                        probabilities = (
                            model.predict_proba(selected_data)[:, 1]
                        )

                        predictions = (
                            probabilities >= threshold
                        ).astype(int)

                        results = pd.DataFrame({
                            "Fraud_Probability": probabilities,
                            "Predicted_Fraud": predictions
                        })

                        results["Risk_Level"] = np.where(
                            probabilities >= 0.90,
                            "High",
                            np.where(
                                predictions == 1,
                                "Medium",
                                "Low"
                            )
                        )

                        results.insert(
                            0,
                            "Transaction_Row",
                            np.arange(1, len(results) + 1)
                        )

                        # --------------------------------------------------
                        # DISPLAY RESULTS
                        # --------------------------------------------------

                        st.subheader("Prediction Results")

                        total_transactions = len(results)

                        fraud_alerts = int(
                            results["Predicted_Fraud"].sum()
                        )

                        genuine_predictions = (
                            total_transactions - fraud_alerts
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Transactions Analyzed",
                            f"{total_transactions:,}"
                        )

                        col2.metric(
                            "Potential Fraud Alerts",
                            f"{fraud_alerts:,}"
                        )

                        col3.metric(
                            "Predicted Genuine",
                            f"{genuine_predictions:,}"
                        )

                        st.divider()

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        # --------------------------------------------------
                        # CHART
                        # --------------------------------------------------

                        st.subheader(
                            "Predicted Risk Distribution"
                        )

                        risk_counts = (
                            results["Risk_Level"]
                            .value_counts()
                            .reindex(
                                ["Low", "Medium", "High"]
                            )
                            .fillna(0)
                            .reset_index()
                        )

                        risk_counts.columns = [
                            "Risk Level",
                            "Transactions"
                        ]

                        fig = px.pie(
                            risk_counts,
                            names="Risk Level",
                            values="Transactions",
                            hole=0.5
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # --------------------------------------------------
                        # DOWNLOAD
                        # --------------------------------------------------

                        st.download_button(
                            "Download Prediction Results",
                            data=results.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="fraudshield_predictions.csv",
                            mime="text/csv"
                        )

                        st.success(
                            "Prediction completed successfully."
                        )

        except Exception as e:

            st.error(
                "An error occurred while processing "
                "the uploaded transaction file."
            )

            st.exception(e)


# --------------------------------------------------
# REAL-TIME FRAUD ALERT SIMULATION
# --------------------------------------------------

elif page == "Real-Time Fraud Alerts":

    st.title("🚨 Real-Time Fraud Alert System")

    st.warning(
        "Demonstration mode: transactions are replayed "
        "from the historical unlabeled test dataset. "
        "This is not a live banking feed."
    )

    try:

        dashboard, daily, hourly = load_dashboard_data()

    except Exception as e:

        st.error("Unable to load dashboard data.")
        st.exception(e)
        st.stop()

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

    if st.button(
        "Process Transactions",
        type="primary"
    ):

        batch = dashboard.iloc[
            int(start_index):
            int(start_index) + batch_size
        ].copy()

        for _, transaction in batch.iterrows():

            probability = transaction[
                "Fraud_Probability"
            ]

            is_alert = (
                transaction["Predicted_Fraud"] == 1
            )

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

        st.dataframe(
            batch,
            use_container_width=True
        )


# --------------------------------------------------
# FRAUD HOTSPOT CALENDAR
# --------------------------------------------------

elif page == "Fraud Hotspot Calendar":

    st.title("📅 Fraud Hotspot Calendar")

    st.write(
        "Explore relative transaction days and hours "
        "with elevated model-predicted fraud activity."
    )

    st.info(
        "Relative day numbers represent elapsed time bins, "
        "not real calendar dates. Hotspots show model "
        "predictions, not confirmed fraud."
    )

    try:

        dashboard, daily, hourly = load_dashboard_data()

    except Exception as e:

        st.error("Unable to load hotspot data.")
        st.exception(e)
        st.stop()

    st.subheader("Daily Predicted Fraud Alerts")

    fig = px.bar(
        daily,
        x="TransactionDay",
        y="Predicted_Alerts",
        color="Alert_Percentage",
        color_continuous_scale="Reds",
        title="Predicted Alerts by Relative Day",
        hover_data=[
            "Total_Transactions",
            "Average_Risk",
            "Alert_Percentage"
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Hourly Predicted Fraud Activity")

    hourly_display = hourly.copy()

    hourly_display["Hour_Label"] = (
        hourly_display["TransactionHour"]
        .astype(str)
        .str.zfill(2)
        + ":00"
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

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Top 10 Relative-Day Hotspots")

    st.dataframe(
        daily.nlargest(
            10,
            "Predicted_Alerts"
        ),
        use_container_width=True
    )


# --------------------------------------------------
# TRANSACTION EXPLORER
# --------------------------------------------------

elif page == "Transaction Explorer":

    st.title("🔎 Transaction Explorer")

    try:

        dashboard, daily, hourly = load_dashboard_data()

    except Exception as e:

        st.error("Unable to load transaction data.")
        st.exception(e)
        st.stop()

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
        & (
            dashboard["Fraud_Probability"]
            >= minimum_probability
        )
    ]

    st.metric(
        "Matching Transactions",
        f"{len(filtered):,}"
    )

    st.dataframe(
        filtered.head(1000),
        use_container_width=True
    )

    st.download_button(
        "Download Filtered Transactions",
        data=filtered.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="fraudshield_filtered_transactions.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "FraudShield AI | CSE303 Capstone Project"
)

st.sidebar.caption(
    "Machine Learning Fraud Detection"
)
