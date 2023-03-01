import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/")
def root():
    """
    Simple route to check server status
    """
    return "OK"


@app.route("/predict", methods=['POST'])
def predict():
    res = request.args
    data = [[res['age'], res['sex'], res['cp'], res['trtbps'], res['chol'], res['fbs'], res['restecg'],
             res['thalachh'], res['exng'], res['oldpeak'], res['slp'], res['caa'], res['thall']]]

    model = joblib.load('data/tree_model.sav')
    pred = model.predict(data)[0]

    return jsonify({'predict': f'{pred}'}), 200


if __name__ == '__main__':
    app.run()
