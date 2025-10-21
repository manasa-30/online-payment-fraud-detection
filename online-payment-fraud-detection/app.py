from flask import Flask, render_template, request, jsonify
from fraud_model import FraudDetector
import os

app = Flask(__name__)
detector = FraudDetector()

# Load model if exists
if os.path.exists('fraud_model.pkl'):
    detector.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = detector.predict(
        data['type'],
        float(data['amount']),
        float(data['oldbalanceOrg']),
        float(data['newbalanceOrig']),
        float(data['oldbalanceDest']),
        float(data['newbalanceDest'])
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)