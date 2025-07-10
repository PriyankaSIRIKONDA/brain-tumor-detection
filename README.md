# 🧠 Brain Tumor Detection using Deep Learning

## 📋 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to automatically detect brain tumors from MRI images. The system can classify brain MRI scans into two categories: **Tumor Present** and **No Tumor**, with the ability to highlight tumor regions when detected.

## 🎯 Key Features

- **Automated Tumor Detection**: Classify brain MRI images as tumor/no-tumor
- **Tumor Highlighting**: Visualize detected tumor regions in images
- **High Accuracy**: Achieves 85%+ accuracy on test data
- **Step-by-Step Implementation**: Modular code structure for easy understanding
- **Comprehensive Analysis**: Confusion matrix, ROC curves, and detailed metrics

## 📊 Dataset

- **Total Images**: 2,990 brain MRI scans
- **Tumor Images**: 1,497 images (labeled as "yes")
- **No Tumor Images**: 1,493 images (labeled as "no")
- **Image Format**: JPG files, grayscale
- **Image Size**: Resized to 128x128 pixels for processing

## 🏗️ Project Structure

```
Brain-Tumor-Detection-Dataset/
├── step1_data_loading.py          # Load and organize dataset
├── step2_preprocessing.py         # Image preprocessing and tumor highlighting
├── step3_cnn_training.py          # CNN model training with augmentation
├── step4_prediction_demo_simple.py # Prediction and visualization
├── step5_model_analysis.py        # Model performance analysis
├── cnn_brain_tumor.pth           # Trained model weights
├── yes/                          # Tumor images folder
├── no/                           # No tumor images folder
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/PriyankaSIRIKONDA/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install required packages**
   ```bash
   pip install torch torchvision
   pip install opencv-python
   pip install numpy matplotlib
   pip install scikit-learn seaborn
   ```

## 📖 Usage

### Step-by-Step Execution

1. **Data Loading** (Step 1)
   ```bash
   python step1_data_loading.py
   ```
   - Loads all images from yes/ and no/ folders
   - Displays dataset statistics
   - Shows sample images

2. **Preprocessing** (Step 2)
   ```bash
   python step2_preprocessing.py
   ```
   - Resizes images to 128x128 pixels
   - Normalizes pixel values
   - Demonstrates tumor highlighting techniques

3. **Model Training** (Step 3)
   ```bash
   python step3_cnn_training.py
   ```
   - Trains CNN model with data augmentation
   - Uses all available images
   - Saves trained model as `cnn_brain_tumor.pth`

4. **Prediction Demo** (Step 4)
   ```bash
   python step4_prediction_demo_simple.py
   ```
   - Tests model on sample images
   - Shows predictions with confidence scores
   - Visualizes results

5. **Model Analysis** (Step 5)
   ```bash
   python step5_model_analysis.py
   ```
   - Generates confusion matrix
   - Plots ROC curve
   - Provides detailed performance metrics

## 🧠 Why CNN for Brain Tumor Detection?

### 1. **Image Processing Capability**
- CNNs are specifically designed for image analysis
- Can automatically learn features from raw pixel data
- No need for manual feature engineering

### 2. **Hierarchical Feature Learning**
- **Low-level features**: Edges, textures, patterns
- **Mid-level features**: Shapes, contours
- **High-level features**: Tumor characteristics, brain structures

### 3. **Spatial Invariance**
- Can detect tumors regardless of their position in the image
- Robust to slight rotations and translations

### 4. **Proven Medical Imaging Success**
- Widely used in medical image analysis
- Excellent performance on MRI, CT, and X-ray images
- FDA-approved for various medical applications

## 🏗️ CNN Architecture

```
Input: 128x128x1 (Grayscale MRI)
├── Conv1: 32 filters, 3x3 kernel → 128x128x32
├── MaxPool: 2x2 → 64x64x32
├── Conv2: 64 filters, 3x3 kernel → 64x64x64
├── MaxPool: 2x2 → 32x32x64
├── Conv3: 128 filters, 3x3 kernel → 32x32x128
├── MaxPool: 2x2 → 16x16x128
├── Flatten → 32,768 features
├── FC1: 128 neurons + ReLU + Dropout(0.25)
└── FC2: 2 neurons (Tumor/No Tumor)
```

## 📈 Model Performance

### Training Results
- **Final Test Accuracy**: 85%
- **Training Accuracy**: 98.44%
- **No Tumor Precision**: 77%
- **No Tumor Recall**: 100%
- **Tumor Precision**: 100%
- **Tumor Recall**: 70%

### Key Insights
- **High Precision for Tumor Detection**: When the model predicts "tumor", it's almost always correct
- **Good Recall for No Tumor**: The model rarely misses healthy brains
- **Balanced Performance**: Good overall accuracy with room for improvement

## 🔧 Technical Details

### Data Augmentation
- **Random Horizontal Flips**: 50% probability
- **Random Vertical Flips**: 50% probability
- **Random Rotations**: 0°, 90°, 180°, 270°
- **Purpose**: Increases dataset size and improves generalization

### Training Parameters
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32
- **Epochs**: 15
- **Train/Test Split**: 80%/20%

### Preprocessing Steps
1. **Resize**: All images to 128x128 pixels
2. **Normalize**: Pixel values to range [0, 1]
3. **Grayscale**: Convert to single channel
4. **Augment**: Apply random transformations during training

## 🎯 Challenges and Solutions

### 1. **Limited Dataset Size**
- **Challenge**: Only ~3,000 images available
- **Solution**: Data augmentation (flips, rotations)
- **Result**: Effective dataset size increased significantly

### 2. **Class Imbalance**
- **Challenge**: Slightly uneven distribution (1497 vs 1493)
- **Solution**: Stratified sampling in train/test split
- **Result**: Balanced representation in both sets

### 3. **Image Quality Variations**
- **Challenge**: Different MRI machines, protocols, quality
- **Solution**: Robust preprocessing and normalization
- **Result**: Consistent input format for the model

### 4. **Overfitting Prevention**
- **Challenge**: Model memorizing training data
- **Solution**: Dropout layers, data augmentation, early stopping
- **Result**: Good generalization to unseen data

## 🔬 Tumor Highlighting Technique

### Method Used
1. **Thresholding**: Apply adaptive threshold to find bright regions
2. **Contour Detection**: Find contours in thresholded image
3. **Filtering**: Remove small contours (noise)
4. **Highlighting**: Draw colored contours around potential tumor regions

### Implementation
```python
def highlight_tumor(img):
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and highlight
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Remove small noise
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    
    return img
```

## 📊 Evaluation Metrics

### Confusion Matrix
- **True Positives (TP)**: Correctly identified tumors
- **True Negatives (TN)**: Correctly identified healthy brains
- **False Positives (FP)**: Healthy brain classified as tumor
- **False Negatives (FN)**: Tumor missed by the model

### ROC Curve
- **AUC Score**: Area Under the Curve
- **Interpretation**: Higher AUC = better model performance
- **Current Performance**: Good separation between classes

## 🚀 Future Improvements

### 1. **Model Architecture**
- Use pre-trained models (ResNet, VGG)
- Implement attention mechanisms
- Add skip connections

### 2. **Data Enhancement**
- Collect more diverse MRI images
- Include different tumor types
- Add metadata (age, gender, tumor stage)

### 3. **Advanced Techniques**
- Implement segmentation (pixel-level tumor detection)
- Use ensemble methods
- Add uncertainty quantification

### 4. **Clinical Integration**
- Real-time processing capabilities
- Integration with PACS systems
- FDA compliance considerations

## 📝 Code Documentation

### Key Functions

#### `load_images(folder, label)`
- **Purpose**: Load images from specified folder
- **Parameters**: folder path, label (0 or 1)
- **Returns**: List of images and labels

#### `preprocess_image(img_path)`
- **Purpose**: Preprocess single image for CNN
- **Parameters**: Image file path
- **Returns**: Normalized 128x128 grayscale image

#### `BrainTumorCNN`
- **Purpose**: CNN architecture for tumor detection
- **Layers**: 3 convolutional + 2 fully connected
- **Activation**: ReLU, Dropout for regularization

#### `predict_single_image(model, image_path, device)`
- **Purpose**: Make prediction on single image
- **Returns**: (prediction, confidence, probabilities)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Priyanka Sirikonda** - Initial work

## 🙏 Acknowledgments

- Dataset providers for brain MRI images
- OpenCV and PyTorch communities
- Medical imaging research community

## 📞 Contact

For questions or support, please open an issue on GitHub or contact the author.

---

**Note**: This project is for educational and research purposes. For clinical use, additional validation and regulatory approval may be required. 