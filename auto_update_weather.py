import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

latitude = 42.625
longitude = 25.1875
timezone = "Europe/Sofia"

weather_file = Path(r"C:\Users\hp pc\Desktop\ThesisProject\data\cleaned_weather.csv")

print("Fetching 14-day forecast from Open-Meteo...")
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "precipitation_probability",
        "wind_speed_10m",
        "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm",
        "soil_moisture_3_to_9cm",
        "soil_moisture_9_to_27cm",
        "soil_temperature_0cm",
        "soil_temperature_6cm",
        "soil_temperature_18cm",
    ],
    "forecast_days": 14,
    "timezone": timezone
}

response = requests.get(url, params=params)
data = response.json()

if "hourly" not in data:
    raise ValueError("No hourly data received from API!")

forecast_df = pd.DataFrame(data["hourly"])
forecast_df["time"] = pd.to_datetime(forecast_df["time"])

forecast_df["latitude"] = latitude
forecast_df["longitude"] = longitude
forecast_df["elevation"] = data.get("elevation", None)
forecast_df["utc_offset_seconds"] = data.get("utc_offset_seconds", None)
forecast_df["timezone"] = timezone
forecast_df["timezone_abbreviation"] = data.get("timezone_abbreviation", None)

if weather_file.exists():
    existing_df = pd.read_csv(weather_file)
    existing_df["time"] = pd.to_datetime(existing_df["time"])
else:
    existing_df = pd.DataFrame()

combined_df = pd.concat([existing_df, forecast_df])
combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values(by="time")

combined_df.to_csv(weather_file, index=False)
print(f"✅ Weather dataset updated — now has {len(combined_df)} records.")
print(f"📂 Saved to: {weather_file}")