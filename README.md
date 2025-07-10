# Brain Tumor Detection with CNN and Grad-CAM

## Overview
This project implements a deep learning pipeline for brain tumor detection in MRI images using a Convolutional Neural Network (CNN) and Grad-CAM for visual explanation. The model classifies images as "tumor" or "no tumor" and highlights the regions most responsible for the decision.

## Dataset Structure
- `yes/` — MRI images with tumors
- `no/` — MRI images without tumors

## Features
- **CNN classifier** for yes/no tumor detection
- **Data augmentation** for robust training
- **Grad-CAM** for visualizing model attention (red/yellow overlay)
- **Realistic evaluation** (no data leakage, honest metrics)
- **Plots:**
  - Training/test loss and accuracy
  - Confusion matrix
  - Grad-CAM overlays on sample predictions

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your MRI images in the `yes/` and `no/` folders.

## Usage
Run the main script:
```bash
python brain_tumor_classifier.py
```

## Outputs
- **Training/test curves:** Show model learning progress
- **Confusion matrix:** Shows true/false positives/negatives
- **Classification report:** Precision, recall, f1-score
- **Grad-CAM overlays:** Red/yellow highlights on tumor predictions

## Model Performance (Example)
- **Accuracy:** ~80%
- **Tumor Recall:** 0.90 (model finds most tumors)
- **No Tumor Precision:** 0.87 (few false alarms)
- **Confidence:** Honest, not overconfident

## How Grad-CAM Works
Grad-CAM visualizes the regions the model focuses on for its decision. For each "tumor" prediction, a red/yellow heatmap is overlaid on the MRI, showing the most influential areas.

## Project Structure
```
Brain-Tumor-Detection-Dataset/
├── yes/                  # Tumor images
├── no/                   # No tumor images
├── brain_tumor_classifier.py  # Main script
├── requirements.txt      # Dependencies
├── README.md             # This file
├── REPORT.md             # Project report
```

## Notes
- This project uses only image-level labels (no pixel masks).
- Grad-CAM provides visual explanation, not precise segmentation.
- For true tumor segmentation, pixel-level masks and a segmentation model are needed. 