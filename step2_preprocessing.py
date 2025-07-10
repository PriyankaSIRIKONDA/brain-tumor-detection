#!/usr/bin/env python3
"""
Step 2: Data Preprocessing and Tumor Highlighting
This is the second step of your brain tumor detection assignment.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("="*60)
print("STEP 2: DATA PREPROCESSING AND TUMOR HIGHLIGHTING")
print("="*60)

# Load image paths (same as step 1)
yes_path = 'yes'
no_path = 'no'
yes_images = [os.path.join(yes_path, img) for img in os.listdir(yes_path) if img.endswith('.jpg')]
no_images = [os.path.join(no_path, img) for img in os.listdir(no_path) if img.endswith('.jpg')]

# Step 2.1: Image Preprocessing Function
print("\n2.1 Creating preprocessing function...")

IMG_SIZE = 128  # Standard size for all images

def preprocess_image(img_path):
    """
    Preprocess a single image:
    1. Convert to grayscale
    2. Resize to standard size
    3. Normalize pixel values to [0,1]
    """
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to standard size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0,1]
    img = img / 255.0
    
    return img

# Test preprocessing on a sample image
print("Testing preprocessing on a sample image...")
sample_img = preprocess_image(yes_images[0])
print(f"Preprocessed image shape: {sample_img.shape}")
print(f"Pixel value range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")

# Step 2.2: Tumor Highlighting Function
print("\n2.2 Creating tumor highlighting function...")

def highlight_tumor(img_path):
    """
    Highlight potential tumor regions in the image using OpenCV
    """
    # Read and preprocess image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Use Otsu's thresholding for automatic threshold selection
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours (boundaries of regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert to color image for highlighting
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Filter contours by area (tumor-like regions)
    min_area = 50   # Minimum area to consider
    max_area = 3000 # Maximum area to consider
    tumor_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    # Draw tumor contours in green
    cv2.drawContours(img_color, tumor_contours, -1, (0, 255, 0), 2)
    
    return img_color, len(tumor_contours), thresh

# Test tumor highlighting
print("Testing tumor highlighting on sample images...")

# Step 2.3: Visualize Preprocessing and Highlighting
print("\n2.3 Visualizing preprocessing and highlighting...")

def visualize_preprocessing_and_highlighting():
    """Show original, preprocessed, and highlighted images"""
    plt.figure(figsize=(15, 10))
    
    # Show examples for tumor images
    for i in range(3):
        if i < len(yes_images):
            # Original image
            plt.subplot(3, 4, i*4 + 1)
            original = cv2.imread(yes_images[i], cv2.IMREAD_GRAYSCALE)
            original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
            plt.imshow(original, cmap='gray')
            plt.title(f'Original Tumor {i+1}')
            plt.axis('off')
            
            # Preprocessed image
            plt.subplot(3, 4, i*4 + 2)
            preprocessed = preprocess_image(yes_images[i])
            plt.imshow(preprocessed, cmap='gray')
            plt.title(f'Preprocessed Tumor {i+1}')
            plt.axis('off')
            
            # Highlighted image
            plt.subplot(3, 4, i*4 + 3)
            highlighted, num_regions, thresh = highlight_tumor(yes_images[i])
            plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
            plt.title(f'Highlighted Tumor {i+1}\n({num_regions} regions)')
            plt.axis('off')
            
            # Threshold image
            plt.subplot(3, 4, i*4 + 4)
            plt.imshow(thresh, cmap='gray')
            plt.title(f'Threshold {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_preprocessing_and_highlighting()
print("Displayed preprocessing and highlighting examples")

# Step 2.4: Prepare Dataset for Machine Learning
print("\n2.4 Preparing dataset for machine learning...")

def prepare_dataset(yes_images, no_images, max_samples=100):
    """
    Prepare the complete dataset with labels for machine learning
    Using max_samples to keep it fast for demonstration
    """
    X = []  # Images
    y = []  # Labels (1 for tumor, 0 for no tumor)
    
    print(f"Processing {max_samples} tumor images...")
    for img_path in tqdm(yes_images[:max_samples]):
        try:
            img = preprocess_image(img_path)
            X.append(img)
            y.append(1)  # Tumor present
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Processing {max_samples} non-tumor images...")
    for img_path in tqdm(no_images[:max_samples]):
        try:
            img = preprocess_image(img_path)
            X.append(img)
            y.append(0)  # No tumor
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y)

# Prepare dataset
X, y = prepare_dataset(yes_images, no_images, max_samples=100)

print(f"Dataset prepared:")
print(f"  - Total images: {X.shape[0]}")
print(f"  - Image size: {X.shape[1]}x{X.shape[2]}")
print(f"  - Tumor samples: {np.sum(y == 1)}")
print(f"  - No tumor samples: {np.sum(y == 0)}")

# Step 2.5: Data Visualization
print("\n2.5 Data visualization...")

plt.figure(figsize=(12, 4))

# Class distribution
plt.subplot(1, 3, 1)
plt.hist(y, bins=2, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Class Distribution')
plt.xlabel('Class (0: No Tumor, 1: Tumor)')
plt.ylabel('Count')
plt.xticks([0, 1])

# Sample images from each class
tumor_indices = np.where(y == 1)[0][:3]
no_tumor_indices = np.where(y == 0)[0][:3]

for i, idx in enumerate(tumor_indices):
    plt.subplot(2, 3, i+2)
    plt.imshow(X[idx], cmap='gray')
    plt.title(f'Tumor {i+1}')
    plt.axis('off')

for i, idx in enumerate(no_tumor_indices):
    plt.subplot(2, 3, i+4)  # Fixed: changed from i+5 to i+4
    plt.imshow(X[idx], cmap='gray')
    plt.title(f'No Tumor {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("STEP 2 COMPLETED SUCCESSFULLY!")
print("="*60)
print(" Image preprocessing function created")
print("Tumor highlighting function created")
print("Preprocessing and highlighting visualized")
print("Dataset prepared for machine learning")
print("Data visualization completed")
print("\nNext: Run step3_feature_engineering.py")
print("="*60) 