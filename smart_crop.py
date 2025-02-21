import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
import json
import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from paho.mqtt import client as mqtt_client

np.random.seed(42)
n_samples = 100

temperature = np.random.uniform(20, 40, n_samples)
humidity = np.random.uniform(40, 90, n_samples)
noise = np.random.normal(0, 5, n_samples)
soil_moisture = 0.5 * humidity + 0.2 * temperature + noise

data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'soil_moisture': soil_moisture
})
print("Sample simulated data for soil moisture model:")
print(data.head())

X = data[['temperature', 'humidity']]
y = data['soil_moisture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
soil_model = RandomForestRegressor(n_estimators=100, random_state=42)
soil_model.fit(X_train, y_train)
y_pred = soil_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on soil moisture test set:", mse)

joblib.dump(soil_model, 'soil_moisture_model.pkl')
print("Soil Moisture Model saved to 'soil_moisture_model.pkl'")
soil_model_loaded = joblib.load('soil_moisture_model.pkl')

if not os.path.exists('fertilizer_recommendation_model.pkl'):
    n_samples_fert = 100
    fert_soil_moisture = np.random.uniform(10, 50, n_samples_fert)
    fert_temperature = np.random.uniform(20, 40, n_samples_fert)
    fert_humidity = np.random.uniform(40, 90, n_samples_fert)
    fert_nitrogen = np.random.uniform(0, 100, n_samples_fert)
    fert_phosphorus = np.random.uniform(0, 100, n_samples_fert)
    fert_potassium = np.random.uniform(0, 100, n_samples_fert)
    fertilizer_types = np.random.choice(['Urea', 'DAP', 'MOP', 'NPK', 'Compost'], n_samples_fert)

    fert_data = pd.DataFrame({
        'soil_moisture': fert_soil_moisture,
        'temperature': fert_temperature,
        'humidity': fert_humidity,
        'nitrogen': fert_nitrogen,
        'phosphorus': fert_phosphorus,
        'potassium': fert_potassium,
        'fertilizer_type': fertilizer_types
    })

    X_fert = fert_data[['soil_moisture', 'temperature', 'humidity', 'nitrogen', 'phosphorus', 'potassium']]
    y_fert = fert_data['fertilizer_type']
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)
    fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fertilizer_model.fit(Xf_train, yf_train)
    y_fert_pred = fertilizer_model.predict(Xf_test)
    accuracy = accuracy_score(yf_test, y_fert_pred)
    print(f"Accuracy of Fertilizer Recommendation Model: {accuracy * 100:.2f}%")
    joblib.dump(fertilizer_model, 'fertilizer_recommendation_model.pkl')
    print("Fertilizer Recommendation Model saved as 'fertilizer_recommendation_model.pkl'")
else:
    fertilizer_model = joblib.load('fertilizer_recommendation_model.pkl')
    print("Fertilizer Recommendation Model loaded.")

ACCESS_TOKEN = "GqXki2ZhhaulT3fwmKLu"
BROKER = "127.0.0.1"
PORT = 1883
TOPIC = "v1/devices/me/telemetry"

def connect_mqtt():
    client = mqtt_client.Client(protocol=mqtt_client.MQTTv311)
    client.username_pw_set(ACCESS_TOKEN)
    client.connect(BROKER, PORT)
    return client

client = connect_mqtt()

csv_file_path = "telemetry_log.csv"
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Temperature", "Humidity", "Predicted Soil Moisture", "Nitrogen", "Phosphorus", "Potassium", "Predicted Fertilizer"])

def log_to_csv(payload):
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            payload["temperature"],
            payload["humidity"],
            payload["predicted_soil_moisture"],
            payload["nitrogen"],
            payload["phosphorus"],
            payload["potassium"],
            payload["predicted_fertilizer"]
        ])

print("Starting continuous sensor simulation...")

current_temp = 25.0
current_humidity = 60.0
current_nitrogen = 50.0
current_phosphorus = 30.0
current_potassium = 100.0

while True:
    current_temp += np.random.normal(0, 0.2)
    current_humidity += np.random.normal(0, 0.5)
    current_nitrogen += np.random.normal(0, 0.1)
    current_phosphorus += np.random.normal(0, 0.1)
    current_potassium += np.random.normal(0, 0.2)

    current_temp = np.clip(current_temp, 20.0, 40.0)
    current_humidity = np.clip(current_humidity, 40.0, 90.0)
    current_nitrogen = np.clip(current_nitrogen, 10.0, 100.0)
    current_phosphorus = np.clip(current_phosphorus, 5.0, 80.0)
    current_potassium = np.clip(current_potassium, 50.0, 200.0)

    sensor_data = pd.DataFrame({'temperature': [current_temp], 'humidity': [current_humidity]})
    predicted_soil_moisture = soil_model_loaded.predict(sensor_data)[0]
    
    fert_features = pd.DataFrame({
        'soil_moisture': [predicted_soil_moisture],
        'temperature': [current_temp],
        'humidity': [current_humidity],
        'nitrogen': [current_nitrogen],
        'phosphorus': [current_phosphorus],
        'potassium': [current_potassium]
    })
    
    predicted_fertilizer_type = fertilizer_model.predict(fert_features)[0]

    payload = {
        "temperature": round(current_temp, 2),
        "humidity": round(current_humidity),
        "predicted_soil_moisture": round(predicted_soil_moisture),
        "nitrogen": round(current_nitrogen),
        "phosphorus": round(current_phosphorus),
        "potassium": round(current_potassium),
        "predicted_fertilizer": predicted_fertilizer_type
    }

    result = client.publish(TOPIC, json.dumps(payload))
    print("Published payload:", payload)
    
    log_to_csv(payload)
    
    time.sleep(2)
