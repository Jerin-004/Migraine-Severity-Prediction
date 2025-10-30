import os
import mlflow
print("Current working directory:", os.getcwd())
print("MLflow tracking URI:", mlflow.get_tracking_uri())
