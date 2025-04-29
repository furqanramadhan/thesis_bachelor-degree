from flask import Flask, jsonify, request
import json
import os
import csv
from urllib.request import urlopen
from datetime import datetime, timezone
import mysql.connector

app = Flask(__name__)

# API Key OpenWeather
API_KEY = "632c374fda93e14b9c8dc45b8e15c900"

# CSV File Path
CSV_SAVE_PATH = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Daily/local/weather_data.csv"

# Database Config
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1903",
    "database": "weather_db",
    "charset": "utf8mb4",
    "collation": "utf8mb4_general_ci"
}

# **Function: Get Weather Data from API**
def fetch_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = urlopen(url)
    return json.loads(response.read().decode("utf-8"))

# **Function: Flatten JSON Data**
def flatten_json(data, parent_key=""):
    items = {}
    for key, value in data.items():
        new_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_json(value, new_key))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            items.update(flatten_json(value[0], new_key))  # Only take first item of list
        else:
            items[new_key] = value
    return items

# **Function: Save Weather Data into MySQL**
def save_weather_data(data):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Field Mapping
        mapped_data = flatten_json(data)
        
        # Database Fields
        fields = ["city", "lat", "lon", "temperature", "feels_like", "humidity", "pressure",
                  "wind_speed", "wind_gust", "wind_deg", "visibility", "cloud_coverage",
                  "weather_main", "weather_description", "rain_1h", "sunrise", "sunset", "timestamp"]

        # Extract & Prepare Data
        values = (
            mapped_data.get("name", "Unknown"),   # city
            mapped_data.get("coord_lat"),
            mapped_data.get("coord_lon"),
            mapped_data.get("main_temp"),
            mapped_data.get("main_feels_like"),
            mapped_data.get("main_humidity"),
            mapped_data.get("main_pressure"),
            mapped_data.get("wind_speed"),
            mapped_data.get("wind_gust"),
            mapped_data.get("wind_deg"),
            mapped_data.get("visibility"),
            mapped_data.get("clouds_all"),
            mapped_data.get("weather_main"),
            mapped_data.get("weather_description"),
            mapped_data.get("rain_1h", 0),  # Default 0 jika tidak ada hujan
            datetime.fromtimestamp(mapped_data.get("sys_sunrise", 0), tz=timezone.utc),
            datetime.fromtimestamp(mapped_data.get("sys_sunset", 0), tz=timezone.utc),
            datetime.now()
        )

        # SQL Insert
        sql = f"INSERT INTO weather_data ({', '.join(fields)}) VALUES ({', '.join(['%s'] * len(fields))})"
        cursor.execute(sql, values)
        conn.commit()

        cursor.close()
        conn.close()
        print("✅ Data berhasil disimpan ke database.")
    except Exception as e:
        print(f"❌ Error menyimpan ke database: {e}")

# **Function: Save Weather Data into CSV**
def save_weather_data_csv(data):
    try:
        mapped_data = flatten_json(data)

        # CSV Header
        fields = ["city", "lat", "lon", "temperature", "feels_like", "humidity", "pressure",
                  "wind_speed", "wind_gust", "wind_deg", "visibility", "cloud_coverage",
                  "weather_main", "weather_description", "rain_1h", "sunrise", "sunset", "timestamp"]

        # Extract & Prepare Data
        values = [
            mapped_data.get("name", "Unknown"),   # city
            mapped_data.get("coord_lat"),
            mapped_data.get("coord_lon"),
            mapped_data.get("main_temp"),
            mapped_data.get("main_feels_like"),
            mapped_data.get("main_humidity"),
            mapped_data.get("main_pressure"),
            mapped_data.get("wind_speed"),
            mapped_data.get("wind_gust"),
            mapped_data.get("wind_deg"),
            mapped_data.get("visibility"),
            mapped_data.get("clouds_all"),
            mapped_data.get("weather_main"),
            mapped_data.get("weather_description"),
            mapped_data.get("rain_1h", 0),  # Default 0 jika tidak ada hujan
            datetime.fromtimestamp(mapped_data.get("sys_sunrise", 0), tz=timezone.utc),
            datetime.fromtimestamp(mapped_data.get("sys_sunset", 0), tz=timezone.utc),
            datetime.now()
        ]

        file_exists = os.path.exists(CSV_SAVE_PATH)

        with open(CSV_SAVE_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write header jika file belum ada
            if not file_exists:
                writer.writerow(fields)
            
            # Write data
            writer.writerow(values)

        print("✅ Data berhasil disimpan ke CSV.")
    except Exception as e:
        print(f"❌ Error menyimpan ke CSV: {e}")

# **API Endpoint: Get Weather Data**
@app.route('/weather', methods=['GET'])
def get_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude dan Longitude diperlukan"}), 400
    
    weather_data = fetch_weather_data(lat, lon)
    
    # Simpan ke Database dan CSV
    save_weather_data(weather_data)
    save_weather_data_csv(weather_data)

    return jsonify(weather_data)

# **API Endpoint: Test Database**
@app.route('/test_db')
def test_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return jsonify({"message": "Koneksi berhasil", "database": db_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
