readme

# Credit Card Fraud Detection System

This project provides a professional pipeline to detect credit card fraud using machine learning. It consists of a training script to build models and a Streamlit web app for interactive fraud detection and analysis.

***

## Features

- Train machine learning models (Logistic Regression, Decision Tree, Random Forest)
- Credit card fraud detection using classic dataset with added features possibility (V1-V8, date, location, transaction metadata)
- Real-time single transaction prediction with confidence and risk level
- Batch prediction from uploaded CSV files
- Explainability using SHAP feature importance plots
- Visualization of transaction location on map
- History of predictions with download option
- Model evaluation plots: confusion matrix, ROC, precision-recall curves
- Detailed report generation with simulated email output to terminal
- User inputs for card holder name, bank name, and card number

***

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone or download the code files:
   - `training.py` for model training
   - `app.py` for Streamlit web app
   - `requirements.txt` for dependencies

2. Install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

***

### Training the Model

Run the training script to download the dataset, train and evaluate models, and save the best model and scaler:

```bash
python training.py
```

The script will save `best_fraud_model.pkl` and `scaler.pkl` to your working directory.

***

### Running the Web Application

After training completes, start the Streamlit app as follows:

```bash
streamlit run app.py
```

This launches an interactive web dashboard where you can:

- Enter single transactions or upload batch CSV files for fraud prediction
- View SHAP explanations for decisions
- Visualize transaction locations on a map
- Review prediction history and download results
- View model performance visualizations
- Generate detailed textual reports and simulate sending via email (print to terminal)

***

### Notes

- The batch CSV file for predictions must include columns: `Time`, `Amount`, and `V1` to `V8`.
- Enter location as `latitude, longitude` in decimal degrees for map visualization.
- Keep `best_fraud_model.pkl` and `scaler.pkl` files in the same directory as `app.py` for proper loading.
- You may expand or customize features and visualizations as needed.

***

### Support

For help with setup, running, or customizing the project, feel free to ask!

***

Thank you for using the Credit Card Fraud Detection System.

الاقتباسات:
