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

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)

# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["Class"].value_counts())

# === Placeholder for additional features ===
# Add columns or process your dataset here for:
# V1 to V8, date, time (seconds), location (lat, lon), previous transaction metadata, bank, card info, etc.

# Separate features and target
X = df.drop("Class", axis=1)
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

    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["ROC AUC"]).sort_values(by="ROC AUC", ascending=False)
    print("\nModel comparison by ROC AUC:\n", results_df)

    # Save best model and scaler
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]
    joblib.dump(best_model, "best_fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(f"Best model saved: {best_model_name}")

if __name__ == "__main__":
    train_and_evaluate()
