import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
from mlflow.models.signature import infer_signature
import os

# --- Ensure relative MLflow directory inside repo ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MLFLOW_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Set MLflow tracking URI ---
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
print("Using MLflow tracking URI:", f"file://{MLFLOW_DIR}")

# --- Load and preprocess data ---
data = pd.read_csv(os.path.join(BASE_DIR, "data", "migraine_data.csv"))
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

X = data.drop('Type', axis=1)
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Start MLflow experiment ---
mlflow.set_experiment("Migraine_Type_Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)

    signature = infer_signature(X_train, model.predict(X_train))

    # ✅ Force model to be saved inside the repo directory only
    artifact_path = os.path.join(MODEL_DIR, "logged_model")
    mlflow.sklearn.save_model(sk_model=model, path=artifact_path, signature=signature)
    mlflow.log_artifacts(MODEL_DIR, artifact_path="model_artifacts")

    print("Model logged successfully in MLflow!")

# --- Save serialized model ---
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
print("✅ Model saved → models/model.pkl")
