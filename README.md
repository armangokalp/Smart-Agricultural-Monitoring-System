# IoT-Driven Climate and Irrigation Control

This project showcases an IoT-based system designed to monitor and optimize plant growth conditions using real-time data and machine learning predictions. By integrating various sensors and automation tools, the system dynamically adjusts soil moisture and temperature parameters to ensure optimal conditions for plants.

## Features
- **Real-Time Monitoring**: Collects data from environmental sensors (temperature, humidity, soil moisture, and light levels) to assess plant conditions.
- **Machine Learning Predictions**: Implements Gradient Boosting and SGD Regressors to forecast temperature and soil moisture for preemptive adjustments.
- **Automated Interventions**: Controls water pumps and fans to maintain optimal conditions based on predictive analytics.
- **Data Visualization**: Includes dashboards to monitor environmental conditions and system decisions.

## Components
- **Hardware**:
  - ESP32 Microcontrollers
  - DHT22 Temperature and Humidity Sensor
  - Soil Moisture Sensor
  - BH1750 Light Sensor
  - Raspberry Pi
- **Software**:
  - Python (for data processing, machine learning, and automation)
  - Matplotlib (for data visualization)
  - Tkinter (for GUI)
  - scikit-learn (for ML models)

## Workflow
1. **Data Collection**: Environmental sensors collect real-time data on temperature, humidity, soil moisture, and light intensity.
2. **Data Processing**: The collected data is cleaned, preprocessed, and used for real-time predictions.
3. **Machine Learning**: Forecasts are generated using pre-trained models to identify trends and potential interventions.
4. **System Control**: Based on predictions, the system activates or deactivates water pumps and fans to maintain ideal conditions.
5. **Monitoring**: Monitoring real-time insights into sensor data and system decisions.

## Figures
- **Figure 1**: Flowerpot with the sensors and components.  
![24 01 2025_01 51 04_REC](https://github.com/user-attachments/assets/53f2099e-7748-45cc-af84-bf15f201d8d4)

- **Figure 2**: Monitoring the environment through the computer.
![24 01 2025_01 51 39_REC](https://github.com/user-attachments/assets/33ef1e1b-f6a8-4df4-b7ee-b62eb805228d)

