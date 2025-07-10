#!/usr/bin/env python3
"""
STEP 3: CNN TRAINING FOR BRAIN TUMOR DETECTION (IMPROVED)
This script loads all images, augments data, trains longer, and prints accuracy.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

print("="*60)
print("STEP 3: CNN TRAINING FOR BRAIN TUMOR DETECTION (IMPROVED)")
print("="*60)

# 1. Load ALL images (not just 200)
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

print("\n1. Loading ALL images...")
yes_imgs, yes_labels = load_images('yes', 1)
no_imgs, no_labels = load_images('no', 0)

X = np.array(yes_imgs + no_imgs)
y = np.array(yes_labels + no_labels)

print(f"Total images: {len(X)} (Tumor: {sum(y)}, No Tumor: {len(y)-sum(y)})")

# 2. Shuffle and split data
def shuffle_split(X, y, test_ratio=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(len(X) * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]

print("\n2. Shuffling and splitting data...")
X_train, y_train, X_test, y_test = shuffle_split(X, y)
print(f"Training set: {len(X_train)} images")
print(f"Testing set: {len(X_test)} images")

# 3. Data augmentation (random flips and rotations)
def augment_image(img):
    # Random horizontal flip
    if random.random() > 0.5:
        img = np.fliplr(img)
    # Random vertical flip
    if random.random() > 0.5:
        img = np.flipud(img)
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        img = np.rot90(img, k=angle//90)
    return img.copy()

class BrainTumorDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx]
        if self.augment:
            img = augment_image(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add channel
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

print("\n3. Creating PyTorch datasets with augmentation...")
train_dataset = BrainTumorDataset(X_train, y_train, augment=True)
test_dataset = BrainTumorDataset(X_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Define a better CNN model
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

print("\n4. Defining CNN architecture...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = BrainTumorCNN().to(device)

# 5. Training the model for more epochs
print("\n5. Training the model (15 epochs, with augmentation)...")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 15
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader.dataset):.4f} - Train Acc: {train_acc:.2%} - Test Acc: {test_acc:.2%}")

# 6. Save the trained model
print("\n6. Saving the trained model...")
torch.save(model.state_dict(), 'cnn_brain_tumor.pth')
print("Model saved as cnn_brain_tumor.pth")

# 7. Plot accuracy curves
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

print("\n" + "="*60)
print("STEP 3 COMPLETED SUCCESSFULLY!")
print("="*60)
print("✅ Model trained with all data and augmentation")
print("✅ Accuracy curves plotted")
print("✅ Model saved for prediction")
print("\nNext: Run step4_prediction_demo_simple.py to test predictions")
print("="*60) 