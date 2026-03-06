from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

import os
model = joblib.load(os.path.join(os.path.dirname(__file__), "fraud_model.pkl"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    claim_amount = float(request.form['claim_amount'])
    claim_frequency = int(request.form['claim_frequency'])
    customer_age = int(request.form['customer_age'])

    data = np.array([[claim_amount, claim_frequency, customer_age]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Fraudulent Claim"
    else:
        result = "Genuine Claim"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":

    app.run(debug=True)
