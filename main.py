import RPi.GPIO as GPIO
import joblib
import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
import socket
import csv
import os

warnings.filterwarnings("ignore")

# ==============================================
# =============== GPIO TANIMLARI ==============
# ==============================================
FAN_PIN = 17
PUMP_PIN = 27

# ==============================================
# ============= MODELLERİ YÜKLEME =============
# ==============================================
temperature_model = joblib.load("temp_model.joblib")
soil_moisture_model = joblib.load("soil_model.joblib")

# ==============================================
# =============== CSV ve UDP AYARLARI =========
# ==============================================
sensor_data_path = "data.csv"
local_ip = "192.168.162.50"
local_port = 12345
buffer_size = 1024

if not os.path.isfile(sensor_data_path):
    with open(sensor_data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Humidity (%)", "Temperature (°C)", "Light (lux)", "Soil Moisture (%)"])
        print(f"Yeni dosya oluşturuldu: {sensor_data_path}")

# ==============================================
# ============ VERİ TOPLAMA FONKSİYONU ========
# ==============================================
def collect_sensor_data(duration=30):
    """
    Belirtilen süre (duration saniye) boyunca UDP'den sensör verilerini dinler
    ve CSV dosyasına yazar. Süre dolunca döner.
    """
    start_time = time.time()

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((local_ip, local_port))
    print(f"[collect_sensor_data] {duration} sn boyunca veri toplanıyor...")

    while True:
        if time.time() - start_time >= duration:
            print("[collect_sensor_data] Süre doldu, veri toplama durduruldu.")
            break

        try:
            data, addr = udp_socket.recvfrom(buffer_size)
            message = data.decode()
            parts = message.split(", ")

            timestamp = parts[0].split(": ")[1]
            humidity = float(parts[1].split(": ")[1].split(" ")[0])
            temperature = float(parts[2].split(": ")[1].split(" ")[0])
            light = float(parts[3].split(": ")[1].split(" ")[0])
            soil_moisture = float(parts[4].split(": ")[1].split(" ")[0])

            with open(sensor_data_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, humidity, temperature, light, soil_moisture])

        except Exception as e:
            print(f"Hata: Veriyi parse edemedim. Hata: {e}")
            continue

    udp_socket.close()

# ==============================================
# ============== TREND ve MÜHENDİSLİK =========
# ==============================================
def calculate_trend(data, column_name, window=3):
    if len(data) < window:
        return 0
    trend = data[column_name].diff().tail(window).mean()
    return trend

def feature_engineering(df):
    df["Temp_lag1"] = df["Temperature (°C)"].shift(1)
    df["Hum_lag1"] = df["Humidity (%)"].shift(1)
    df["Soil_lag1"] = df["Soil Moisture (%)"].shift(1)
    df["Light_lag1"] = df["Light (lux)"].shift(1)
    df.dropna(inplace=True)
    return df

# ==============================================
# ========== TAHMİN FONKSİYONU (N STEPS) =======
# ==============================================
def forecast_n_steps(model, latest_data, steps=500):
    feature_columns = [
        "Temperature (°C)", "Humidity (%)", "Light (lux)", "Soil Moisture (%)",
        "Temp_lag1","Hum_lag1","Light_lag1","Soil_lag1"
    ]
    forecast_data = pd.DataFrame(latest_data, columns=feature_columns).values

    for step in range(steps):
        prediction = model.predict(forecast_data)[0]
        forecast_data[0][0] = prediction

    return prediction

# ==============================================
# ============ FAN ve POMPA KONTROLÜ ==========
# ==============================================
def control_fan_dynamic(temperature_forecast, temperature_trend):
    if temperature_trend > 0 and temperature_forecast > 25:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(FAN_PIN, GPIO.OUT)
        GPIO.output(FAN_PIN, GPIO.HIGH)
        print(f"Fan çalıştırıldı (Trend pozitif + Forecast > 25). Tahmin {temperature_forecast:.2f} °C")
        time.sleep(15)
        GPIO.cleanup()
        print("Fan kapatıldı.")
    elif temperature_forecast > 30:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(FAN_PIN, GPIO.OUT)
        GPIO.output(FAN_PIN, GPIO.HIGH)
        print(f"Fan çalıştırıldı (Forecast > 30). Tahmin {temperature_forecast:.2f} °C")
        time.sleep(15)
        GPIO.cleanup()
        print("Fan kapatıldı.")
    else:
        print(f"Sıcaklık normal, fan kapalı kalacak. Tahmin {temperature_forecast:.2f} °C")

def control_pump_dynamic(soil_moisture_forecast, soil_moisture_trend):
    if soil_moisture_trend < 0 and soil_moisture_forecast < 40:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PUMP_PIN, GPIO.OUT)
        GPIO.output(PUMP_PIN, GPIO.HIGH)
        print(f"Pompa çalıştırıldı (Trend negatif + Forecast < 40). Nem {soil_moisture_forecast:.2f} %")
        time.sleep(3)
        GPIO.cleanup()
        print("Pompa kapatıldı.")
    elif soil_moisture_forecast < 30:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PUMP_PIN, GPIO.OUT)
        GPIO.output(PUMP_PIN, GPIO.HIGH)
        print(f"Pompa çalıştırıldı (Forecast < 30). Nem {soil_moisture_forecast:.2f} %")
        time.sleep(3)
        GPIO.cleanup()
        print("Pompa kapatıldı.")
    else:
        print(f"Toprak nemi normal, pompa kapalı kalacak. Tahmin {soil_moisture_forecast:.2f} %")

# ==============================================
# =============== MAIN LOOP ====================
# ==============================================
if __name__ == "__main__":
    try:
        while True:
            collect_sensor_data(duration=30)

            sensor_data = pd.read_csv(sensor_data_path)
            sensor_data = feature_engineering(sensor_data)

            if len(sensor_data) == 0:
                print("Yeterli veri yok. Tekrar veri toplanacak.")
                continue

            latest_data = sensor_data.iloc[[-1]][[
                "Temperature (°C)", "Humidity (%)", "Light (lux)", "Soil Moisture (%)",
                "Temp_lag1","Hum_lag1","Light_lag1","Soil_lag1"
            ]].values

            temperature_forecast = forecast_n_steps(temperature_model, latest_data, steps=500)
            soil_moisture_forecast = forecast_n_steps(soil_moisture_model, latest_data, steps=500)

            temperature_trend = calculate_trend(sensor_data, "Temperature (°C)")
            soil_moisture_trend = calculate_trend(sensor_data, "Soil Moisture (%)")

            control_fan_dynamic(temperature_forecast, temperature_trend)
            control_pump_dynamic(soil_moisture_forecast, soil_moisture_trend)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] 500 adım sonrası tahmin: Temp={temperature_forecast:.2f}°C, Soil={soil_moisture_forecast:.2f}%")

            time.sleep(10)

    except KeyboardInterrupt:
        print("Program durduruldu. GPIO temizleniyor...")
        GPIO.cleanup()
