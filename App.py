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

# Initialize prediction history
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(
        columns=[
            "Timestamp", "Card Holder", "Bank", "Card Number",
            "Date", "Time", "Location", "Amount",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
            "Prediction", "Fraud Probability", "Risk Level"
        ]
    )

st.sidebar.title("Settings")

mode = st.sidebar.radio("Prediction Mode", ["Single Transaction", "Batch Prediction"])

show_insights = st.sidebar.checkbox("Show Model Insights", value=True)

# Card information inputs
st.sidebar.subheader("Card Information")
card_holder_name = st.sidebar.text_input("Card Holder Name")
bank_name = st.sidebar.text_input("Bank Name")
card_number = st.sidebar.text_input("Card Number")

st.title("💳 Credit Card Fraud Detection System")

def compute_risk(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

def display_prediction(prediction, probability):
    risk = compute_risk(probability)
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction!\nConfidence: {probability:.2f}, Risk Level: {risk}")
    else:
        st.success(f"✅ Legitimate Transaction\nConfidence: {1 - probability:.2f}, Risk Level: {risk}")

def add_to_history(entry):
    st.session_state["history"] = pd.concat([st.session_state["history"], entry], ignore_index=True)

def show_history():
    st.subheader("Prediction History")
    if not st.session_state["history"].empty:
        hist_df = st.session_state["history"].sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(hist_df)
        csv_export = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", data=csv_export, file_name="prediction_history.csv")
    else:
        st.info("No prediction history yet.")

if mode == "Single Transaction":
    st.subheader("Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        date_val = st.date_input("Date")
        time_val = st.time_input("Time")
        amount_val = st.number_input("Amount", min_value=0.0, format="%.2f")

    with col2:
        location_val = st.text_input("Location (latitude, longitude)")
        v_features = []
        for i in range(1, 9):
            val = st.number_input(f"V{i}", value=0.0, format="%.4f")
            v_features.append(val)

    if st.button("Predict"):
        seconds = time_val.hour * 3600 + time_val.minute * 60 + time_val.second
        features = np.array([[seconds, amount_val] + v_features])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = (model.predict_proba(features_scaled)[:, 1][0]
                       if hasattr(model, "predict_proba")
                       else model.decision_function(features_scaled)[0])

        display_prediction(prediction, probability)

        # SHAP explainability
        explainer = shap.Explainer(model, features_scaled)
        shap_values = explainer(features_scaled)
        st.subheader("Prediction Explanation (SHAP)")
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(plt)

        # Show location on map
        if location_val:
            try:
                lat, lon = map(float, location_val.split(","))
                st.subheader("Transaction Location")
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
            except Exception:
                st.warning("Invalid location format, please enter as 'latitude, longitude'")

        # Save to history
        entry = pd.DataFrame([{
            "Timestamp": datetime.now(),
            "Card Holder": card_holder_name,
            "Bank": bank_name,
            "Card Number": card_number,
            "Date": date_val.strftime("%Y-%m-%d"),
            "Time": time_val.strftime("%H:%M:%S"),
            "Location": location_val,
            "Amount": amount_val,
            **{f"V{i}": v for i, v in enumerate(v_features, 1)},
            "Prediction": prediction,
            "Fraud Probability": probability,
            "Risk Level": compute_risk(probability),
        }])
        add_to_history(entry)

else:
    st.subheader("Batch Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 9)]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
        else:
            df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

            preds = model.predict(df[required_cols])
            probs = (model.predict_proba(df[required_cols])[:, 1]
                     if hasattr(model, "predict_proba")
                     else model.decision_function(df[required_cols]))

            df["Prediction"] = preds
            df["Fraud Probability"] = probs
            df["Risk Level"] = df["Fraud Probability"].apply(compute_risk)

            st.write("Batch Prediction Results")
            st.dataframe(df.head())

            batch_history = df.copy()
            batch_history["Timestamp"] = datetime.now()
            batch_history["Card Holder"] = card_holder_name
            batch_history["Bank"] = bank_name
            batch_history["Card Number"] = card_number
            add_to_history(batch_history)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Batch Predictions CSV", data=csv_bytes, file_name="batch_predictions.csv")

show_history()

if show_insights:
    st.subheader("Model Performance Insights (Demo Data)")

    # Confusion Matrix (placeholder)
    cm = np.array([[56780, 10], [20, 80]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve (demo)
    fpr, tpr, _ = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # Precision-Recall Curve (demo)
    precision, recall, _ = precision_recall_curve([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision, marker=".")
    ax3.set_title("Precision-Recall Curve")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    st.pyplot(fig3)

st.subheader("Generate Detailed Report and Simulate Email Output")

if st.button("Generate Report"):
    if st.session_state["history"].empty:
        st.warning("No prediction history available for report.")
    else:
        last_entry = st.session_state["history"].iloc[-1]
        features_list = [f"V{i}={last_entry['V' + str(i)]:.4f}" for i in range(1, 9)]
        features_str = ', '.join(features_list)
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
Features: {features_str}
Prediction: {"Fraudulent" if last_entry['Prediction'] == 1 else "Legitimate"}
Confidence: {last_entry['Fraud Probability']:.4f}
Risk Level: {last_entry['Risk Level']}
"""
        st.text_area("Generated Report", report, height=300)
        st.text("Simulated Email Output (view in terminal/log):")
        print("\n=== Fraud Detection Report Email ===\n")
        print(report)
