from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_FILE = os.path.join(MODEL_DIR, "label_encoders.pkl")
DATA_FILE = "dataset/patient_data.csv"

# -----------------------------
# Load dataset
# -----------------------------
def load_data():
    return pd.read_csv(DATA_FILE)

# -----------------------------
# Preprocess data
# -----------------------------
def preprocess_data(df):
    df = df.dropna()
    target_column = 'Stages'  # Correct target column

    label_encoders = {}
    categorical_values = {}

    for col in df.select_dtypes(include='object').columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            categorical_values[col] = le.classes_.tolist()

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders, categorical_values, list(X.columns)

# -----------------------------
# Train model
# -----------------------------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

# -----------------------------
# Save artifacts
# -----------------------------
def save_artifacts(model, scaler, label_encoders):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoders, ENCODERS_FILE)

# -----------------------------
# Flask routes
# -----------------------------
@app.route('/')
def home():
    df = load_data()
    _, _, _, _, categorical_values, input_columns = preprocess_data(df)
    return render_template('index.html', categorical_values=categorical_values, input_columns=input_columns)

@app.route('/predict', methods=['POST'])
def predict():
    df_input = pd.DataFrame([request.form.to_dict()])

    # Load model, scaler, encoders
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)

    # Encode categorical inputs
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col].astype(str))

    # Convert numeric inputs
    for col in df_input.columns:
        df_input[col] = pd.to_numeric(df_input[col], errors='ignore')

    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled).max() * 100

    return render_template('result.html', prediction=prediction, probability=probability)

# -----------------------------
# Main execution
# -----------------------------
if __name__ == '__main__':
    df = load_data()
    X, y, scaler, label_encoders, categorical_values, input_columns = preprocess_data(df)
    model = train_model(X, y)
    save_artifacts(model, scaler, label_encoders)
    app.run(debug=True)
