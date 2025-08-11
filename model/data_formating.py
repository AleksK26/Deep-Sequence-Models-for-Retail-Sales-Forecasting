import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class TomatoWeatherDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load the merged dataset
        self.data = pd.read_csv(csv_file)

        # Store transforms
        self.transform = transform

        # Extract class names
        self.classes = sorted(self.data["stage"].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Define which columns are weather features
        self.weather_cols = [
            col for col in self.data.columns
            if col not in ["file_path", "stage", "image_datetime"]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row
        row = self.data.iloc[idx]

        # Load image
        img_path = row["file_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Weather features → Tensor
        weather_data = torch.tensor(row[self.weather_cols].values, dtype=torch.float32)

        # Label as integer
        label = self.class_to_idx[row["stage"]]

        return image, weather_data, label
