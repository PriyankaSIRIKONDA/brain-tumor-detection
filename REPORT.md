# Brain Tumor Detection Project Report

## Objective
To build a deep learning pipeline for classifying brain MRI images as "tumor" or "no tumor" and visually highlight the regions responsible for the model's decision using Grad-CAM.

## Data
- MRI images in two folders: `yes/` (tumor), `no/` (no tumor)
- Only image-level labels (no pixel masks)

## Methods
- **CNN classifier**: Simple, regularized architecture
- **Data augmentation**: Random flips for robust training
- **Train/test split**: 75% train, 25% test (no data leakage)
- **Grad-CAM**: Visual explanation of model predictions

## Training & Evaluation
- **Loss/accuracy curves**: Show steady improvement, no overfitting
- **Confusion matrix**:
  - True Negatives: 267
  - False Positives: 111
  - False Negatives: 39
  - True Positives: 333
- **Classification report**:
  - No Tumor: Precision 0.87, Recall 0.71, F1 0.78
  - Tumor: Precision 0.75, Recall 0.90, F1 0.82
  - Accuracy: 0.80
- **Confidence analysis**:
  - Average: 0.70
  - Correct: 0.73
  - Wrong: 0.59

## Visual Outputs
- **Training/test loss and accuracy plots**
- **Confusion matrix heatmap**
- **Grad-CAM overlays**: Red/yellow highlights on "tumor" predictions, showing model attention

## Key Findings
- The model achieves realistic, honest performance (no overfitting, no data leakage)
- Grad-CAM provides meaningful visual explanations, but not precise segmentation
- For true tumor localization, pixel-level masks and a segmentation model would be required

## Limitations
- Only image-level labels available
- Grad-CAM is a visual explanation, not a segmentation
- Some false positives/negatives remain

## Conclusion
This project demonstrates a robust, interpretable pipeline for brain tumor detection in MRI images using only yes/no labels. The model is honest, generalizes well, and provides visual explanations for its decisions. 