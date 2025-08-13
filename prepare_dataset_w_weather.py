import os 
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Weather db reformatting
weather_csv = Path(r"C:\Users\hp pc\Desktop\ThesisProject\data\March_June25Weather.csv")

with open(weather_csv, "r", encoding="utf-8") as f:
    location_header = f.readline().strip()
    location_values = f.readline().strip()

location_info = dict(zip(location_header.split(","), location_values.split(",")))
print("Location Info:", location_info)

weather_df = pd.read_csv(weather_csv, skiprows=2, header=0)

weather_df = weather_df[weather_df["time"] != "time"]

weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce")  # auto detect

weather_df = weather_df.dropna(subset=["time"])

for key, value in location_info.items():
    weather_df[key] = value

print("\nWeather Data Preview:")
print(weather_df.head())

output_path = Path(r"C:\Users\hp pc\Desktop\ThesisProject\data\cleaned_weather.csv")
weather_df.to_csv(output_path, index=False)

print(f"\nCleaned weather data saved to {output_path}")


image_folder = r"data\Tomato_Plant_Stages_Dataset"
weather_new_csv = r"data\cleaned_weather.csv"
output_csv = r"data\tomato_dataset_with_weather.csv"

stage_date_range = {
    'Stage1_Early_Vegetative': ("2025-03-25", "2025-05-05"),
    'Stage2_Flowering_Initiation': ("2025-05-25", "2025-06-30")
}

weather_df = pd.read_csv(weather_new_csv)

weather_df['time'] = pd.to_datetime(weather_df['time'])

# Goal is to assign sequential dates so the data is more accurate for training
data_records = []

for stage, (start_date_str, end_date_str) in stage_date_range.items():
    stage_path = os.path.join(image_folder, stage)
    if not os.path.exists(stage_path):
        print(f"Stage folder not fpund: {stage_path}")
        continue

    image_files = sorted([f for f in os.listdir(stage_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Calculate days between images
    total_days = (end_date - start_date).days
    if len(image_files) > 1:
        days_between = total_days / (len(image_files) - 1)
    else:
        days_between = 0

    for idx, img_file in enumerate(image_files):
        assigned_date = start_date + timedelta(days=round(idx * days_between))

        # Find closest weather record
        weather_row = weather_df.iloc[(weather_df["time"] - assigned_date).abs().argsort()[:1]]

        if not weather_row.empty:
            record = {
                "image_path": os.path.join(stage_path, img_file),
                "stage": stage,
                "date": assigned_date.date(),
            }
            # Merge weather columns
            for col in weather_df.columns:
                if col != "time":
                    record[col] = weather_row.iloc[0][col]

            data_records.append(record)

final_df = pd.DataFrame(data_records)
final_df.to_csv(output_csv, index=False)

print(f"Dataset with sequential dates & weather saved to {output_csv}")
print(final_df.head())