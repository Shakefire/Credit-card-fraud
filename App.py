import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Initialize session state history
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(
        columns=[
            "Timestamp", "Card Holder", "Bank", "Card Number",
            "Date", "Time", "Location", "Amount", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
            "Prediction", "Fraud Probability", "Risk Level"
        ]
    )

st.sidebar.title("Settings")

# Model selection (for this example, only best saved model is used)
model_options = ["Best Model"]
selected_model_name = st.sidebar.selectbox("Select Model for Prediction", model_options)

mode = st.sidebar.radio("Prediction Mode", ["Single Transaction", "Batch Prediction"])

show_insights = st.sidebar.checkbox("Show Model Insights", value=True)

# Card details inputs
st.sidebar.subheader("Card Information")
card_holder_name = st.sidebar.text_input("Card Holder Name")
bank_name = st.sidebar.text_input("Bank Name")
card_number = st.sidebar.text_input("Card Number")

# Main title
st.title("💳 Credit Card Fraud Detection System")

def compute_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

def show_prediction_results(pred, prob):
    risk = compute_risk_level(prob)
    if pred == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\nConfidence: {prob:.2f}, Risk Level: {risk}")
    else:
        st.success(f"✅ Legitimate Transaction\nConfidence: {1 - prob:.2f}, Risk Level: {risk}")

def add_history_entry(data):
    st.session_state["history"] = pd.concat([st.session_state["history"], data], ignore_index=True)

def display_history():
    st.subheader("Prediction History")
    if not st.session_state["history"].empty:
        st.dataframe(st.session_state["history"].sort_values(by="Timestamp", ascending=False).reset_index(drop=True))
        csv = st.session_state["history"].to_csv(index=False).encode("utf-8")
        st.download_button("Download History as CSV", data=csv, file_name="prediction_history.csv")
    else:
        st.info("No prediction history yet.")

if mode == "Single Transaction":
    st.subheader("Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Date")
        time = st.time_input("Time")
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")

    with col2:
        location = st.text_input("Location (latitude, longitude)")
        v_features = []
        for i in range(1, 9):
            v = st.number_input(f"V{i}", value=0.0, format="%.4f")
            v_features.append(v)

    if st.button("Predict"):
        # Convert time to seconds from midnight
        total_seconds = time.hour * 3600 + time.minute * 60 + time.second

        # Prepare input feature vector
        input_features = np.array([[total_seconds, amount] + v_features])
        input_scaled = scaler.transform(input_features)

        # Prediction
        pred = model.predict(input_scaled)[0]
        prob = (
            model.predict_proba(input_scaled)[:, 1][0]
            if hasattr(model, "predict_proba")
            else model.decision_function(input_scaled)[0]
        )

        show_prediction_results(pred, prob)

        # SHAP explainability
        explainer = shap.Explainer(model, input_scaled)
        shap_values = explainer(input_scaled)
        st.subheader("Explanation (SHAP Feature Importance)")
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(plt)

        # Location map visualization
        if location:
            try:
                lat, lon = map(float, location.split(","))
                st.subheader("Transaction Location")
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
            except Exception:
                st.warning("Invalid location format. Use: latitude, longitude")

        # Save prediction to history
        entry = pd.DataFrame([{
            "Timestamp": datetime.now(),
            "Card Holder": card_holder_name,
            "Bank": bank_name,
            "Card Number": card_number,
            "Date": date.strftime("%Y-%m-%d"),
            "Time": time.strftime("%H:%M:%S"),
            "Location": location,
            "Amount": amount,
            **{f"V{i}": v for i, v in enumerate(v_features, 1)},
            "Prediction": pred,
            "Fraud Probability": prob,
            "Risk Level": compute_risk_level(prob),
        }])
        add_history_entry(entry)

elif mode == "Batch Prediction":
    st.subheader("Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)

        # Basic validation and scaling
        required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 9)]
        missing_cols = [c for c in required_cols if c not in df_batch.columns]
        if missing_cols:
            st.error(f"Missing columns in CSV: {missing_cols}")
        else:
            # Scale Time and Amount
            df_batch[["Time", "Amount"]] = scaler.transform(df_batch[["Time", "Amount"]])

            # Prediction
            preds = model.predict(df_batch[required_cols])
            probs = (
                model.predict_proba(df_batch[required_cols])[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(df_batch[required_cols])
            )
            df_batch["Prediction"] = preds
            df_batch["Fraud Probability"] = probs
            df_batch["Risk Level"] = df_batch["Fraud Probability"].apply(compute_risk_level)

            st.write("Prediction Results:")
            st.dataframe(df_batch.head())

            # Save batch predictions to history
            batch_history = df_batch.copy()
            batch_history["Timestamp"] = datetime.now()
            batch_history["Card Holder"] = card_holder_name
            batch_history["Bank"] = bank_name
            batch_history["Card Number"] = card_number
            add_history_entry(batch_history.rename(columns={"Risk Level": "Risk Level"}))

            csv = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", data=csv, file_name="batch_predictions.csv")

# Display History
display_history()

# Model Insights Visualization
if show_insights:
    st.subheader("Model Performance Insights")
    # Confusion Matrix (placeholder with dummy data)
    dummy_cm = np.array([[56780, 10], [20, 80]])
    fig, ax = plt.subplots()
    sns.heatmap(dummy_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Demo)")
    st.pyplot(fig)

    # ROC Curve (demo)
    fpr, tpr, _ = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_title("ROC Curve (Demo)")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # Precision-Recall Curve (demo)
    precision, recall, _ = precision_recall_curve([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision, marker=".")
    ax3.set_title("Precision-Recall Curve (Demo)")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    st.pyplot(fig3)

# Report generation & Email simulation

st.subheader("Generate Report and Simulate Email Output")

if st.button("Generate Report"):
    # Compose report string from last prediction in history
    if st.session_state["history"].empty:
        st.warning("No prediction in history to generate report.")
    else:
        last_entry = st.session_state["history"].iloc[-1]
        report = f"""
        Credit Card Fraud Detection Report
        -----------------------------------
        Timestamp: {last_entry['Timestamp']}
        Card Holder: {last_entry['Card Holder']}
        Bank: {last_entry['Bank']}
        Card Number: {last_entry['Card Number']}
        Date: {last_entry['Date']}
        Time: {last_entry['Time']}
        Location: {last_entry['Location']}
        Amount: {last_entry['Amount']}
        Features: {', '.join([f'V{i}={last_entry[f\"V{i}\"]:.4f}' for i in range(1, 9)])}
        Prediction: {"Fraudulent" if last_entry['Prediction'] == 1 else "Legitimate"}
        Confidence: {last_entry['Fraud Probability']:.4f}
        Risk Level: {last_entry['Risk Level']}
        """
        st.text_area("Report", report, height=300)

        st.text("Simulated Email Output (displayed in terminal):")
        print("\n=== Fraud Detection Report Email ===\n")
        print(report)
