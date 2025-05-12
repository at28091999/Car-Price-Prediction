from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("CarPricePredictLRModel.pkl", "rb"))

# Load data
df = pd.read_csv("Clean Car Dataset.csv")

# Dropdown base lists
companies = sorted(df["company"].dropna().unique())
years = sorted(df["year"].dropna().unique(), reverse=True)
fuel_types = sorted(df["fuel_type"].dropna().unique())

@app.route("/", methods=["GET", "POST"])
def index():
    selected_company = None
    car_names = []

    if request.method == "POST":
        selected_company = request.form.get("company")
        car_names = sorted(df[df["company"] == selected_company]["name"].dropna().unique())

        if "predict" in request.form:
            name = request.form["name"]
            year = int(request.form["year"])
            kms_driven = int(request.form["kms_driven"])
            fuel_type = request.form["fuel_type"]

            input_df = pd.DataFrame(
                [[name, selected_company, year, kms_driven, fuel_type]],
                columns=["name", "company", "year", "kms_driven", "fuel_type"]
            )

            prediction = model.predict(input_df)[0]
            return render_template(
                "index.html",
                companies=companies,
                years=years,
                fuel_types=fuel_types,
                car_names=car_names,
                selected_company=selected_company,
                selected_name=name,
                prediction=round(prediction, 2)
            )

    return render_template(
        "index.html",
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        car_names=car_names,
        selected_company=selected_company
    )

if __name__ == "__main__":
    app.run(debug=True)
