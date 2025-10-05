# Signature Recognition Using CNN and Traditional Feature Extraction

A comprehensive comparison of deep learning (CNN) and traditional computer vision approaches for handwritten signature recognition and classification.

## 🎯 Project Overview

This project implements and compares different approaches for signature recognition:

1. **Deep Learning Approach**: Convolutional Neural Network (CNN)
2. **Traditional Computer Vision**: HOG + SVM/Random Forest
3. **Traditional Computer Vision**: LBP + SVM/Random Forest  
4. **Traditional Computer Vision**: ORB + SVM/Random Forest

## 📊 Dataset

The project uses the [Signature Verification Dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) from Kaggle, which contains signature images from multiple individuals.

## 🚀 Features

- **Data Loading & Preprocessing**: Automated dataset loading with multiple path detection
- **CNN Implementation**: Custom CNN architecture with data augmentation
- **Feature Extraction**: HOG, LBP, and ORB feature extraction methods
- **Multiple Classifiers**: SVM and Random Forest for traditional approaches
- **Comprehensive Evaluation**: Precision, Recall, F1-Score, Accuracy metrics
- **Visualization**: Training curves, confusion matrices, performance comparisons
- **Error Analysis**: Detailed insights and method comparisons

## 📋 Requirements

```python
tensorflow>=2.8.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🛠️ Installation & Usage

### For Kaggle Notebooks:
1. Add the [Signature Verification Dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) to your notebook
2. Copy and run the provided code cells
3. The dataset will be automatically detected at `/kaggle/input/signature-verification-dataset`

### For Google Colab:
1. Upload the dataset to your Google Drive
2. Uncomment the drive mounting code in Cell 2
3. Update the `dataset_path` variable to point to your dataset location
4. Run all cells sequentially

### For Local Environment:
```bash
pip install -r requirements.txt
jupyter notebook signature_recognition_project.ipynb
```


## 🔍 Methodology

### 1. Data Preprocessing
- Image resizing to 128×128 pixels
- Grayscale conversion
- Normalization (0-1 range)
- Train/Validation/Test split (60/20/20)

### 2. CNN Architecture
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
→ Flatten → Dropout(0.5) → Dense(128) → Dropout(0.5) → Dense(num_classes)
```

### 3. Traditional Feature Extraction
- **HOG (Histogram of Oriented Gradients)**: 9 orientations, 8×8 pixels per cell
- **LBP (Local Binary Patterns)**: Radius=3, 24 points, uniform patterns
- **ORB (Oriented FAST and Rotated BRIEF)**: 500 keypoints maximum

### 4. Classifiers
- **Support Vector Machine (SVM)**: RBF kernel
- **Random Forest**: 100 estimators

## 📈 Results

The project generates comprehensive performance metrics:

- **Accuracy Scores** for all method combinations
- **Precision, Recall, F1-Score** for each class
- **Confusion Matrices** for visual performance assessment
- **Training Curves** for CNN model analysis
- **Comparative Analysis** between all approaches

### Sample Performance (typical results):
```
Method Comparison:
├── CNN: ~85-95% accuracy
├── HOG + SVM: ~75-85% accuracy  
├── LBP + Random Forest: ~70-80% accuracy
└── ORB + SVM: ~65-75% accuracy
```

## 📊 Visualizations

The project generates multiple visualization types:

1. **Sample Images**: Display of signature samples from different classes
2. **Training Curves**: CNN accuracy and loss over epochs
3. **Confusion Matrices**: Performance breakdown by class
4. **Performance Comparison**: Bar charts comparing all methods
5. **Feature Visualization**: HOG and LBP feature representations

## 🔧 Customization

### Adjusting CNN Architecture:
```python
def create_custom_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Add your custom layers here
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        # ... more layers
    ])
    return model
```

### Adding New Feature Extractors:
```python
def extract_custom_features(images):
    features = []
    for img in images:
        # Your feature extraction logic
        feature_vector = your_feature_function(img)
        features.append(feature_vector)
    return np.array(features)
```

## 🐛 Troubleshooting

### Common Issues:

1. **"No data loaded" error**: 
   - Ensure dataset is properly uploaded
   - Check dataset path and folder structure
   - Verify image file formats (.png, .jpg, .jpeg)

2. **Memory errors**:
   - Reduce batch size in CNN training
   - Limit number of ORB features
   - Use smaller image dimensions

3. **Training slow/fails**:
   - Reduce number of epochs
   - Use smaller dataset subset for testing
   - Enable GPU acceleration in Colab/Kaggle

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{signature_recognition_2024,
  title={Signature Recognition Using CNN and Traditional Feature Extraction},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/signature-recognition}}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **Sarmad Razaq** - *Initial work* - (https://github.com/SarmadRazaq)

## 🙏 Acknowledgments

- Kaggle for providing the signature verification dataset
- TensorFlow and scikit-learn communities for excellent documentation
- Computer vision researchers for developing HOG, LBP, and ORB algorithms

---

⭐ **Star this repository if you found it helpful!**
