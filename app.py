from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("CarPricePredictLRModel.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    company = request.form["company"]
    year = int(request.form["year"])
    kms_driven = int(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]

    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                            columns=["name", "company", "year", "kms_driven", "fuel_type"])
    
    prediction = model.predict(input_df)[0]
    return render_template("index.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
