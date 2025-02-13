from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            month = int(request.form["month"])
            dew_point = float(request.form["dew_point"])
            humidity = float(request.form["humidity"])
            wind_speed = float(request.form["wind_speed"])
            precipitation = float(request.form["precipitation"])

            input_data = np.array([[month, dew_point, humidity, wind_speed, precipitation]])
            
            feature_names = ['Month', 'DewPointAvgF', 'HumidityAvgPercent', 'WindAvgMPH', 'PrecipitationSumInches']
            input_df = pd.DataFrame(input_data, columns=feature_names)
            
            input_scaled = scaler.transform(input_df)
            
            prediction = model.predict(input_scaled)[0]

            return render_template("index.html", prediction=prediction)
        except:
            return render_template("index.html", error="Invalid input.")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
