import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import time as dtime

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)

# ==============================
# Training Section
# ==============================

# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["Class"].value_counts())

# Feature selection (match Streamlit UI)
selected_features = ["Time", "Amount"] + [f"V{i}" for i in range(1, 9)]
X = df[selected_features]
y = df["Class"]

# Scale Time and Amount
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=30, max_depth=8, class_weight="balanced", random_state=42
    ),
}

results = {}

def train_and_evaluate():
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        auc_score = roc_auc_score(y_test, y_proba)
        results[name] = auc_score
        print(f"{name} ROC AUC: {auc_score:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

    results_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["ROC AUC"]
    ).sort_values(by="ROC AUC", ascending=False)
    print("\nModel comparison by ROC AUC:\n", results_df)

    # Save best model and scaler
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]
    joblib.dump(best_model, "best_fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(f"Best model saved: {best_model_name}")

# Train and save artifacts
train_and_evaluate()

# ==============================
# Prediction Section
# ==============================

# Load best model and scaler
model = joblib.load("best_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

def compute_risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

def predict_transaction(time_obj, amount, v_features):
    """
    Predict fraud for a single transaction.
    Parameters:
        time_obj : datetime.time
        amount   : float
        v_features : list of length 8 (V1–V8 values)
    Returns:
        dict with Prediction, Fraud Probability, Risk Level
    """
    if len(v_features) != 8:
        raise ValueError("v_features must contain exactly 8 values (V1–V8).")

    # Convert time to seconds
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

    # Scale only Time and Amount
    scaled_time_amount = scaler.transform([[total_seconds, amount]])[0]

    # Combine scaled (Time, Amount) with raw V1–V8
    input_features = np.array([list(scaled_time_amount) + list(v_features)])

    # Prediction
    pred = model.predict(input_features)[0]
    prob = (
        model.predict_proba(input_features)[:, 1][0]
        if hasattr(model, "predict_proba")
        else model.decision_function(input_features)[0]
    )

    return {
        "Prediction": "Fraudulent" if pred == 1 else "Legitimate",
        "Fraud Probability": round(prob, 4),
        "Risk Level": compute_risk_level(prob),
    }

# ==============================
# Example usage
# ==============================

if __name__ == "__main__":
    example = predict_transaction(
        time_obj=dtime(14, 35, 20),
        amount=250.00,
        v_features=[0.1, -1.2, 0.5, 2.3, -0.7, 1.1, 0.0, -0.5]
    )
    print("\nExample prediction:", example)￼Enter
