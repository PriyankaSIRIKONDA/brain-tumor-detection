#!/usr/bin/env python3
"""
Step 4: Prediction Demo - Using the Trained CNN Model (Simplified)
This script shows how to use your trained CNN to predict tumor/no-tumor on new images.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("="*60)
print("STEP 4: PREDICTION DEMO - USING TRAINED CNN MODEL")
print("="*60)

# Step 4.1: Define the CNN model (same as in step 3)
print("\n4.1 Defining CNN model...")

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

def preprocess_image(img_path):
    """Preprocess a single image for CNN"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

# Step 4.2: Load the trained model
print("\n4.2 Loading the trained model...")

# Check if model file exists
model_path = 'cnn_brain_tumor.pth'
if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found!")
    print("Please run step3_cnn_training.py first to train the model.")
    exit()

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model loaded successfully!")

# Step 4.3: Create prediction function
print("\n4.3 Creating prediction function...")

def predict_single_image(model, image_path, device):
    """
    Predict tumor/no-tumor for a single image
    Returns: (prediction, confidence, probabilities)
    """
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

print("✅ Prediction function created!")

# Step 4.4: Test predictions on sample images
print("\n4.4 Testing predictions on sample images...")

# Load some test images
yes_path = 'yes'
no_path = 'no'
yes_images = [os.path.join(yes_path, img) for img in os.listdir(yes_path) if img.endswith('.jpg')]
no_images = [os.path.join(no_path, img) for img in os.listdir(no_path) if img.endswith('.jpg')]

# Select sample images for testing
test_tumor_images = yes_images[:3]    # First 3 tumor images
test_no_tumor_images = no_images[:3]  # First 3 no-tumor images

print(f"Testing on {len(test_tumor_images)} tumor images and {len(test_no_tumor_images)} no-tumor images")

# Make predictions
tumor_results = []
for img_path in test_tumor_images:
    try:
        result = predict_single_image(model, img_path, device)
        tumor_results.append(result)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        tumor_results.append(None)

no_tumor_results = []
for img_path in test_no_tumor_images:
    try:
        result = predict_single_image(model, img_path, device)
        no_tumor_results.append(result)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        no_tumor_results.append(None)

# Step 4.5: Visualize predictions
print("\n4.5 Visualizing predictions...")

plt.figure(figsize=(15, 8))

# Plot tumor images and their predictions
for i, (img_path, result) in enumerate(zip(test_tumor_images, tumor_results)):
    if result is not None:
        pred, conf, probs = result
        
        # Load and display image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        
        plt.subplot(2, 6, i+1)
        plt.imshow(img, cmap='gray')
        
        # Color code based on prediction accuracy
        if pred == 1:  # Correctly predicted as tumor
            title_color = 'green'
            prediction_text = 'Tumor ✓'
        else:  # Incorrectly predicted as no tumor
            title_color = 'red'
            prediction_text = 'No Tumor ✗'
        
        plt.title(f'{prediction_text}\nConf: {conf:.3f}', 
                  color=title_color, fontweight='bold', fontsize=10)
        plt.axis('off')

# Plot no-tumor images and their predictions
for i, (img_path, result) in enumerate(zip(test_no_tumor_images, no_tumor_results)):
    if result is not None:
        pred, conf, probs = result
        
        # Load and display image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        
        plt.subplot(2, 6, i+7)
        plt.imshow(img, cmap='gray')
        
        # Color code based on prediction accuracy
        if pred == 0:  # Correctly predicted as no tumor
            title_color = 'green'
            prediction_text = 'No Tumor ✓'
        else:  # Incorrectly predicted as tumor
            title_color = 'red'
            prediction_text = 'Tumor ✗'
        
        plt.title(f'{prediction_text}\nConf: {conf:.3f}', 
                  color=title_color, fontweight='bold', fontsize=10)
        plt.axis('off')

plt.tight_layout()
plt.show()

# Step 4.6: Print detailed results
print("\n4.6 Detailed prediction results:")

print("\nTumor Images (should be predicted as Tumor):")
print("-" * 50)
correct_tumor = 0
for i, (img_path, result) in enumerate(zip(test_tumor_images, tumor_results)):
    if result is not None:
        pred, conf, probs = result
        status = "✓ CORRECT" if pred == 1 else "✗ WRONG"
        if pred == 1:
            correct_tumor += 1
        print(f"Image {i+1}: Predicted {'Tumor' if pred==1 else 'No Tumor'} "
              f"(Confidence: {conf:.3f}) - {status}")

print(f"\nNo-Tumor Images (should be predicted as No Tumor):")
print("-" * 50)
correct_no_tumor = 0
for i, (img_path, result) in enumerate(zip(test_no_tumor_images, no_tumor_results)):
    if result is not None:
        pred, conf, probs = result
        status = "✓ CORRECT" if pred == 0 else "✗ WRONG"
        if pred == 0:
            correct_no_tumor += 1
        print(f"Image {i+1}: Predicted {'Tumor' if pred==1 else 'No Tumor'} "
              f"(Confidence: {conf:.3f}) - {status}")

# Calculate accuracy
total_correct = correct_tumor + correct_no_tumor
total_images = len(test_tumor_images) + len(test_no_tumor_images)
accuracy = total_correct / total_images

print(f"\n" + "="*50)
print(f"SUMMARY:")
print(f"Tumor images correctly classified: {correct_tumor}/{len(test_tumor_images)}")
print(f"No-tumor images correctly classified: {correct_no_tumor}/{len(test_no_tumor_images)}")
print(f"Overall accuracy: {accuracy:.2%}")
print("="*50)

# Step 4.7: Test on a single image with detailed output
print("\n4.7 Testing on a single image with detailed output...")

if len(test_tumor_images) > 0:
    test_img = test_tumor_images[0]
    pred, conf, probs = predict_single_image(model, test_img, device)
    
    print(f"\nDetailed Analysis for: {test_img}")
    print(f"Predicted: {'Tumor' if pred==1 else 'No Tumor'}")
    print(f"Confidence: {conf:.3f}")
    print(f"Probability No Tumor: {probs[0]:.3f}")
    print(f"Probability Tumor: {probs[1]:.3f}")
    
    # Show the image with prediction
    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Input Image\n{os.path.basename(test_img)}')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    bars = plt.bar(['No Tumor', 'Tumor'], probs, color=['blue', 'red'])
    plt.title('Prediction Probabilities')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{prob:.3f}', ha='center', va='bottom')
    
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.8, f'Prediction: {"Tumor" if pred==1 else "No Tumor"}', 
             fontsize=14, fontweight='bold', color='green' if conf > 0.7 else 'orange')
    plt.text(0.1, 0.6, f'Confidence: {conf:.3f}', fontsize=12)
    plt.text(0.1, 0.4, f'No Tumor Prob: {probs[0]:.3f}', fontsize=12)
    plt.text(0.1, 0.2, f'Tumor Prob: {probs[1]:.3f}', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*60)
print("STEP 4 COMPLETED SUCCESSFULLY!")
print("="*60)
print("✅ Model loaded and tested")
print("✅ Predictions visualized")
print("✅ Accuracy calculated")
print("✅ Detailed analysis completed")
print("\nNext: Run step5_model_analysis.py for advanced analysis")
print("="*60) 