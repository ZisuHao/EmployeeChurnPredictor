from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        Age = request.form["Age"]
        MonthlyIncome = request.form["MonthlyIncome"]
        TotalEarnInCompany = request.form["TotalEarnInCompany"]
        X = np.array([[float(Age), float(MonthlyIncome), float(TotalEarnInCompany)]])
        pred = model.predict_proba(X)[0][1]
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
