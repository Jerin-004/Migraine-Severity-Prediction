from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("./models/model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array(features).reshape(1, -1)
    prediction = model.predict(final_input)[0]
    return render_template('index.html', prediction_text=f"Predicted Migraine Type: {prediction}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
