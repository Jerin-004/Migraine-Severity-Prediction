import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
from mlflow.models.signature import infer_signature

data = pd.read_csv("./data/migraine_data.csv")

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

X = data.drop('Type', axis=1)
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Migraine_Type_Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)

    # Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model with name, signature, and input example (to avoid warnings)
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train.iloc[:1]
    )
    print("Model logged in MLflow successfully!")

joblib.dump(model, "./models/model.pkl")
print("Model saved → models/model.pkl")
