from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open("CarPricePredictLRModel.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset to get dropdown options
df = pd.read_csv("Clean Car Dataset.csv")

# Unique sorted dropdown options
car_names = sorted(df["name"].dropna().unique())
companies = sorted(df["company"].dropna().unique())
years = sorted(df["year"].dropna().unique(), reverse=True)
fuel_types = sorted(df["fuel_type"].dropna().unique())

@app.route("/")
def index():
    return render_template(
        "index.html",
        car_names=car_names,
        companies=companies,
        years=years,
        fuel_types=fuel_types
    )

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    company = request.form["company"]
    year = int(request.form["year"])
    kms_driven = int(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]

    input_df = pd.DataFrame(
        [[name, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        car_names=car_names,
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        prediction=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
