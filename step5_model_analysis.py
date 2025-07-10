#!/usr/bin/env python3
"""
STEP 5: MODEL ANALYSIS (CONFUSION MATRIX & ROC CURVE)
This script loads the trained model, tests it on the test set, and shows a confusion matrix and ROC curve.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

print("="*60)
print("STEP 5: MODEL ANALYSIS (CONFUSION MATRIX & ROC CURVE)")
print("="*60)

# 1. Load test data (same as in training)
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

yes_imgs, yes_labels = load_images('yes', 1)
no_imgs, no_labels = load_images('no', 0)
X = np.array(yes_imgs + no_imgs)
y = np.array(yes_labels + no_labels)

# Use the same split as before
def shuffle_split(X, y, test_ratio=0.2):
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(len(X) * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]

_, _, X_test, y_test = shuffle_split(X, y)

class BrainTumorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx] / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

test_dataset = BrainTumorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. Load the trained model
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load('cnn_brain_tumor.pth', map_location=device))
model.eval()

# 3. Get predictions and probabilities
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of tumor
        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 4. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['No Tumor', 'Tumor']))

# 5. ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print(f"ROC AUC: {roc_auc:.2f}")

print("\n" + "="*60)
print("STEP 5 COMPLETED SUCCESSFULLY!")
print("="*60)
print("✅ Confusion matrix and ROC curve plotted")
print("✅ Model performance analyzed")
print("\nNext: Update prediction script to highlight tumors!")
print("="*60) 