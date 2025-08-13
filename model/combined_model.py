import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b0
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TomatoWeatherDataset(Dataset):
    """Custom dataset that combines images with weather features and days to harvest"""
    
    def __init__(self, csv_file, transform=None, use_weather=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_weather = use_weather
        
        # Get class names and create label mapping
        self.class_names = sorted(self.data['stage'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Define weather feature columns (based on your dataset)
        self.weather_features = [
            'temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)',
            'soil_temperature_0_to_7cm (°C)', 'soil_temperature_7_to_28cm (°C)',
            'soil_moisture_0_to_7cm (m³/m³)', 'soil_moisture_7_to_28cm (m³/m³)',
            'wind_speed_10m (km/h)'
        ]
        
        # Filter only existing columns
        self.weather_features = [col for col in self.weather_features if col in self.data.columns]
        
        if self.use_weather and len(self.weather_features) > 0:
            # Normalize weather features
            self.scaler = StandardScaler()
            weather_data = self.data[self.weather_features].values
            self.weather_normalized = self.scaler.fit_transform(weather_data)
            
            # Save scaler for later use
            joblib.dump(self.scaler, 'weather_scaler.pkl')
            print(f"📊 Using {len(self.weather_features)} weather features")
        else:
            self.weather_normalized = None
            print("⚠️ Not using weather features")
        
        # Assume a fixed harvest date (adjust based on variety/climate; ~129 days from start)
        self.harvest_date = pd.to_datetime('2025-08-01')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        stage = self.data.iloc[idx]['stage']
        label = self.class_to_idx[stage]
        
        # Calculate days to harvest
        date = pd.to_datetime(self.data.iloc[idx]['date'])
        days_to_harvest = max(0, (self.harvest_date - date).days)
        
        # Get weather features if available
        if self.use_weather and self.weather_normalized is not None:
            weather = torch.FloatTensor(self.weather_normalized[idx])
            return image, weather, label, days_to_harvest
        else:
            return image, label, days_to_harvest

class MultiModalTomatoNet(nn.Module):
    """Neural network that combines CNN for images with MLP for weather data, with multi-task outputs"""
    
    def __init__(self, num_classes, num_weather_features=0, use_weather=True):
        super(MultiModalTomatoNet, self).__init__()
        self.use_weather = use_weather and num_weather_features > 0
        
        # Image feature extractor (EfficientNet)
        self.image_encoder = efficientnet_b0(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in self.image_encoder.features.parameters():
            param.requires_grad = False
        
        # Get the number of features from EfficientNet
        image_features = self.image_encoder.classifier[1].in_features
        
        # Remove the original classifier
        self.image_encoder.classifier = nn.Identity()
        
        if self.use_weather:
            # Weather feature processor
            self.weather_encoder = nn.Sequential(
                nn.Linear(num_weather_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # Combined features
            combined_features = image_features + 64
        else:
            combined_features = image_features
        
        # Shared layers for combined features
        self.shared = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head (for stage)
        self.classifier = nn.Linear(256, num_classes)
        
        # Regression head (for days to harvest)
        self.regressor = nn.Linear(256, 1)
    
    def forward(self, image, weather=None):
        # Process image
        image_features = self.image_encoder(image)
        
        if self.use_weather and weather is not None:
            # Process weather data
            weather_features = self.weather_encoder(weather)
            # Combine features
            combined = torch.cat([image_features, weather_features], dim=1)
        else:
            combined = image_features
        
        # Shared layers
        shared_out = self.shared(combined)
        
        # Outputs
        class_out = self.classifier(shared_out)
        reg_out = self.regressor(shared_out)
        return class_out, reg_out

def train_enhanced_model():
    """Train the enhanced model with weather integration and multi-task learning"""
    
    # Configuration
    dataset_csv = "data/tomato_dataset_with_weather.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_csv):
        print(f"❌ Dataset not found: {dataset_csv}")
        print("Please run prepare_dataset_w_weather.py first")
        return
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = TomatoWeatherDataset(dataset_csv, transform=transform_train, use_weather=True)
    
    print(f"📊 Dataset loaded: {len(full_dataset)} samples")
    print(f"🏷️ Classes: {full_dataset.class_names}")
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = transform_val
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"🚂 Train samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Model setup
    num_weather_features = len(full_dataset.weather_features) if full_dataset.weather_normalized is not None else 0
    model = MultiModalTomatoNet(
        num_classes=len(full_dataset.class_names),
        num_weather_features=num_weather_features,
        use_weather=True
    )
    model = model.to(device)
    
    print(f"🧠 Model created with {num_weather_features} weather features")
    
    # Training setup
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')
    train_class_losses, train_reg_losses, val_class_losses, val_reg_losses = [], [], [], []
    train_accs, val_accs, val_maes = [], [], []
    
    print("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_class_loss, train_reg_loss, train_correct = 0.0, 0.0, 0
        
        for batch in tqdm(train_loader, desc="Training"):
            if len(batch) == 4:  # Image, weather, label, days
                images, weather, labels, days = batch
                images, weather, labels, days = images.to(device), weather.to(device), labels.to(device), torch.tensor(days).to(device).float().unsqueeze(1)
                class_out, reg_out = model(images, weather)
            else:  # Fallback (no weather)
                images, labels, days = batch
                images, labels, days = images.to(device), labels.to(device), torch.tensor(days).to(device).float().unsqueeze(1)
                class_out, reg_out = model(images)
            
            optimizer.zero_grad()
            loss_class = criterion_class(class_out, labels)
            loss_reg = criterion_reg(reg_out, days)
            loss = loss_class + loss_reg  # Equal weight; tune if needed
            loss.backward()
            optimizer.step()
            
            train_class_loss += loss_class.item() * images.size(0)
            train_reg_loss += loss_reg.item() * images.size(0)
            _, preds = torch.max(class_out, 1)
            train_correct += torch.sum(preds == labels.data)
        
        # Validation phase
        model.eval()
        val_class_loss, val_reg_loss, val_correct, val_mae = 0.0, 0.0, 0, 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if len(batch) == 4:  # Image, weather, label, days
                    images, weather, labels, days = batch
                    images, weather, labels, days = images.to(device), weather.to(device), labels.to(device), torch.tensor(days).to(device).float().unsqueeze(1)
                    class_out, reg_out = model(images, weather)
                else:
                    images, labels, days = batch
                    images, labels, days = images.to(device), labels.to(device), torch.tensor(days).to(device).float().unsqueeze(1)
                    class_out, reg_out = model(images)
                
                loss_class = criterion_class(class_out, labels)
                loss_reg = criterion_reg(reg_out, days)
                
                val_class_loss += loss_class.item() * images.size(0)
                val_reg_loss += loss_reg.item() * images.size(0)
                _, preds = torch.max(class_out, 1)
                val_correct += torch.sum(preds == labels.data)
                val_mae += torch.mean(torch.abs(reg_out - days)).item() * images.size(0)
        
        # Calculate metrics
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        train_class_loss /= n_train
        train_reg_loss /= n_train
        val_class_loss /= n_val
        val_reg_loss /= n_val
        train_acc = train_correct.double() / n_train
        val_acc = val_correct.double() / n_val
        val_mae /= n_val
        val_total_loss = val_class_loss + val_reg_loss
        
        train_class_losses.append(train_class_loss)
        train_reg_losses.append(train_reg_loss)
        val_class_losses.append(val_class_loss)
        val_reg_losses.append(val_reg_loss)
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        val_maes.append(val_mae)
        
        # Update learning rate
        scheduler.step(val_total_loss)
        
        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'class_names': full_dataset.class_names,
                'num_classes': len(full_dataset.class_names),
                'weather_features': full_dataset.weather_features,
                'use_weather': True
            }, "best_enhanced_model.pth")
            print(f"💾 Best model saved! Val Loss: {val_total_loss:.4f}")
        
        print(f"📈 Train Class Loss: {train_class_loss:.4f}, Reg Loss: {train_reg_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"📊 Val Class Loss: {val_class_loss:.4f}, Reg Loss: {val_reg_loss:.4f}, Acc: {val_acc:.4f}, MAE (days): {val_mae:.4f}")
        print(f"🎯 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_class_losses, label='Train Class Loss')
    plt.plot(val_class_losses, label='Val Class Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_reg_losses, label='Train Reg Loss')
    plt.plot(val_reg_losses, label='Val Reg Loss')
    plt.title('Regression Loss (Days to Harvest)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.plot(val_maes, label='Val MAE (days)')
    plt.title('Accuracy and MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_enhanced_model()