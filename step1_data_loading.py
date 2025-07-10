#!/usr/bin/env python3
"""
Step 1: Data Loading and Basic Setup
This is the first step of your brain tumor detection assignment.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("="*60)
print("STEP 1: DATA LOADING AND BASIC SETUP")
print("="*60)

# Step 1.1: Check your dataset structure
print("\n1.1 Checking dataset structure...")
yes_path = 'yes'
no_path = 'no'

if not os.path.exists(yes_path):
    print(f" Directory '{yes_path}' not found!")
    exit()
if not os.path.exists(no_path):
    print(f"Directory '{no_path}' not found!")
    exit()

print(f"Found '{yes_path}' directory")
print(f"Found '{no_path}' directory")

# Step 1.2: List all image files
print("\n1.2 Listing image files...")
yes_images = [os.path.join(yes_path, img) for img in os.listdir(yes_path) if img.endswith('.jpg')]
no_images = [os.path.join(no_path, img) for img in os.listdir(no_path) if img.endswith('.jpg')]

print(f"Found {len(yes_images)} tumor images (yes/)")
print(f"Found {len(no_images)} non-tumor images (no/)")

if len(yes_images) == 0 or len(no_images) == 0:
    print("❌ No images found! Check your dataset.")
    exit()

# Step 1.3: Check if images can be loaded
print("\n1.3 Checking image validity...")

def is_image_valid(img_path):
    """Check if an image can be loaded properly"""
    try:
        img = cv2.imread(img_path)
        return img is not None
    except:
        return False

# Test a few images
print("Testing first few images...")
for i, img_path in enumerate(yes_images[:3]):
    if is_image_valid(img_path):
        print(f" Tumor image {i+1}: OK")
    else:
        print(f" Tumor image {i+1}: Corrupted")

for i, img_path in enumerate(no_images[:3]):
    if is_image_valid(img_path):
        print(f" Non-tumor image {i+1}: OK")
    else:
        print(f" Non-tumor image {i+1}: Corrupted")

# Step 1.4: Show sample images
print("\n1.4 Displaying sample images...")

def show_sample_images(image_list, title, num_samples=3):
    """Display sample images from a list"""
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(image_list))):
        img = cv2.imread(image_list[i], cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{title} {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show sample tumor images
show_sample_images(yes_images, "Tumor")
print("Displayed sample tumor images")

# Show sample non-tumor images  
show_sample_images(no_images, "No Tumor")
print("Displayed sample non-tumor images")

print("\n" + "="*60)
print("STEP 1 COMPLETED SUCCESSFULLY!")
print("="*60)
print("Dataset structure verified")
print("Image files listed")
print("Image validity checked")
print("Sample images displayed")
print("\nNext: Run step2_preprocessing.py")
print("="*60) 