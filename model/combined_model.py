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
    """Custom dataset that combines images with weather features"""
    
    def __init__(self, csv_file, transform=None, use_weather=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_weather = use_weather
        
        # Get class names and create label mapping
        self.class_names = sorted(self.data['stage'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Weather feature columns
        self.weather_features = [col for col in self.data.columns if col.startswith('weather_')]
        
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
        
        # Get weather features if available
        if self.use_weather and self.weather_normalized is not None:
            weather = torch.FloatTensor(self.weather_normalized[idx])
            return image, weather, label
        else:
            return image, label

class MultiModalTomatoNet(nn.Module):
    """Neural network that combines CNN for images with MLP for weather data"""
    
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
            
            # Combined classifier
            combined_features = image_features + 64
        else:
            combined_features = image_features
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
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
        
        # Final classification
        output = self.classifier(combined)
        return output

def train_enhanced_model():
    """Train the enhanced model with weather integration"""
    
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for batch in tqdm(train_loader, desc="Training"):
            if len(batch) == 3:  # Image, weather, label
                images, weather, labels = batch
                images, weather, labels = images.to(device), weather.to(device), labels.to(device)
                outputs = model(images, weather)
            else:  # Image, label only
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if len(batch) == 3:  # Image, weather, label
                    images, weather, labels = batch
                    images, weather, labels = images.to(device), weather.to(device), labels.to(device)
                    outputs = model(images, weather)
                else:  # Image, label only
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        # Calculate metrics
        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        val_acc = val_correct.double() / len(val_dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': full_dataset.class_names,
                'num_classes': len(full_dataset.class_names),
                'weather_features': full_dataset.weather_features,
                'use_weather': True
            }, "best_enhanced_model.pth")
            print(f"💾 Best model saved! Accuracy: {val_acc:.4f}")
        
        print(f"📈 Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"🎯 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Feature importance could be added here if needed
    plt.bar(range(len(full_dataset.weather_features)), 
            np.random.random(len(full_dataset.weather_features)))  # Placeholder
    plt.title('Weather Feature Usage')
    plt.xticks(range(len(full_dataset.weather_features)), 
               [f.replace('weather_', '') for f in full_dataset.weather_features], 
               rotation=45)
    plt.tight_layout()
    
    plt.savefig('enhanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_enhanced_model()