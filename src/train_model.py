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

# Paths (use relative, not absolute)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "migraine_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MLRUNS_DIR = os.path.join(ROOT_DIR, "mlruns")

# Ensure safe local directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# Set MLflow tracking to a local directory (relative)
mlflow.set_tracking_uri("file://" + MLRUNS_DIR)
mlflow.set_experiment("Migraine_Type_Prediction")

# Load dataset
data = pd.read_csv(DATA_PATH)

# Convert categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col].astype(str))

# Split dataset
X = data.drop("Type", axis=1)
y = data["Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and log to MLflow
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
        signature=signature,
        input_example=X_train.iloc[:1],
    )

    print(f"✅ Model logged successfully! Accuracy: {acc:.4f}")

# Save local model file
model_path = os.path.join(MODEL_DIR, "model.pkl")
joblib.dump(model, model_path)
print(f"💾 Model saved → {model_path}")
