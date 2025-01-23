import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(csv_path="data_initial.csv"):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df = df[["Timestamp","Temperature (°C)","Humidity (%)","Light (lux)","Soil Moisture (%)"]]
    return df

def feature_engineering_temp(df):
    df["TempTarget"] = df["Temperature (°C)"].shift(-1)
    df["Temp_lag1"] = df["Temperature (°C)"].shift(1)
    df["Hum_lag1"] = df["Humidity (%)"].shift(1)
    df["Soil_lag1"] = df["Soil Moisture (%)"].shift(1)
    df["Light_lag1"] = df["Light (lux)"].shift(1)
    df.dropna(inplace=True)
    return df

def feature_engineering_soil(df):
    df["SoilTarget"] = df["Soil Moisture (%)"].shift(-1)
    df["Soil_lag1"] = df["Soil Moisture (%)"].shift(1)
    df["Temp_lag1"] = df["Temperature (°C)"].shift(1)
    df["Hum_lag1"] = df["Humidity (%)"].shift(1)
    df["Light_lag1"] = df["Light (lux)"].shift(1)
    df.dropna(inplace=True)
    return df

def train_temperature_model(df):
    y = df["TempTarget"]
    X = df[[
        "Temperature (°C)","Humidity (%)","Light (lux)","Soil Moisture (%)",
        "Temp_lag1","Hum_lag1","Soil_lag1","Light_lag1"
    ]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("=== Temperature Model ===")
    print("MSE:", mean_squared_error(y_te, pred))
    print("R2 :", r2_score(y_te, pred))
    joblib.dump(model, "temp_model.joblib")
    return model

def train_soil_model(df):
    y = df["SoilTarget"]
    X = df[[
        "Temperature (°C)","Humidity (%)","Light (lux)","Soil Moisture (%)",
        "Soil_lag1","Temp_lag1","Hum_lag1","Light_lag1"
    ]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print("=== Soil Moisture Model ===")
    print("MSE:", mean_squared_error(y_te, pred))
    print("R2 :", r2_score(y_te, pred))
    joblib.dump(model, "soil_model.joblib")
    return model

def predict_temperature(model, sample):
    cols = [
        "Temperature (°C)","Humidity (%)","Light (lux)","Soil Moisture (%)",
        "Temp_lag1","Hum_lag1","Soil_lag1","Light_lag1"
    ]
    return model.predict(sample[cols])

def predict_soil_moisture(model, sample):
    cols = [
        "Temperature (°C)","Humidity (%)","Light (lux)","Soil Moisture (%)",
        "Soil_lag1","Temp_lag1","Hum_lag1","Light_lag1"
    ]
    return model.predict(sample[cols])

if __name__ == "__main__":
    df = load_data("data_initial.csv")

    temp_df = df.copy()
    temp_df = feature_engineering_temp(temp_df)
    temp_model = train_temperature_model(temp_df)

    soil_df = df.copy()
    soil_df = feature_engineering_soil(soil_df)
    soil_model = train_soil_model(soil_df)

    if len(temp_df) > 0 and len(soil_df) > 0:
        temp_sample = temp_df.iloc[[-1]].copy()
        soil_sample = soil_df.iloc[[-1]].copy()

        next_temp = predict_temperature(temp_model, temp_sample)[0]
        next_soil = predict_soil_moisture(soil_model, soil_sample)[0]

        print("\n--- Decision Logic ---")
        print(f"Next Temp: {next_temp:.2f} °C | Next Soil: {next_soil:.2f} %")

        fan = "ON" if next_temp > 30 else "OFF"
        pump = "ON" if next_soil < 30 else "OFF"

        print(f"Fan: {fan}, Pump: {pump}")
