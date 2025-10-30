FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn mlflow

COPY . .

RUN python ./src/train_model.py

EXPOSE 8080
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8080", "app.app:app"]
