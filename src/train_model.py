import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib

# Force everything to stay inside the repo
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "migraine_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MLRUNS_DIR = os.path.join(ROOT_DIR, "mlruns")

# Clean and ensure correct directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# Force MLflow to use a local repo path
mlflow_tracking_uri = f"file://{MLRUNS_DIR}"
print("Using MLflow tracking URI:", mlflow_tracking_uri)
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Migraine_Type_Prediction")

# Load dataset
data = pd.read_csv(DATA_PATH)

# Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col].astype(str))

X = data.drop("Type", axis=1)
y = data["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train + log
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)

    signature = infer_signature(X_train, model.predict(X_train.iloc[:1]))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=None,  # local logging only
        signature=signature,
        input_example=X_train.iloc[:1]
    )

    print(f"✅ Model logged successfully! Accuracy: {acc:.4f}")

# Save model to local folder
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, model_path)
print(f"💾 Model saved → {model_path}")
