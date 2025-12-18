# cnn-and-efficientnet-on-dental-dataset
https://www.kaggle.com/code/soultanyousif/cnn-and-efficientnet-on-dental-dataset


This report analyzes the performance of Deep Learning models on the Tooth dataset for classifying dental conditions. Two architectures were evaluated: a Custom CNN and a Pretrained EfficientNetB0. The study demonstrates that Transfer Learning (EfficientNetB0) significantly outperforms the custom architecture, achieving a test accuracy of 97.76% compared to 93.80% for the CNN.
## Dataset Overview
- **Source**: Tooth dataset ([/kaggle/input/teeth-dataset/Tooth dataset](https://www.kaggle.com/datasets/rajapriyanshu/teeth-dataset))
- **Classes**: 5 categories
  - Calculus
  - Caries
  - Hypodontia
  - Mouth Ulcer
  - Tooth Discoloration
- **Data Distribution Strategy**: 
  - Stratified Split: 70% Train, 15% Validation, 15% Test.
  - Class Imbalance Handling: Computed class weights (balanced mode, capped at 3.0) to give more importance to underrepresented classes like 'caries'.

## Methodology

### 1. Data Preprocessing
- **Resize**: Images resized to 224x224 pixels.
- **Augmentation**: Random flip, brightness (0.1), contrast (0.9-1.1), and saturation adjustments applied to training data.
- **Normalization**: 
  - CNN: Rescaled to [0,1].
  - EfficientNet: ImageNet specific preprocessing.

### 2. Model Architectures

#### Model A: Custom CNN
- **Structure**: 4 Convolutional Blocks (Conv2D -> BN -> ReLU -> Pooling -> Dropout).
- **Classifier**: GlobalAveragePooling -> Dense(256) -> Dropout -> Output(Softmax).
- **Training**: Adam optimizer (lr=0.0005), 100 epochs, Early Stopping.

#### Model B: EfficentNetB0 (Transfer Learning)
- **Base**: Pretrained on ImageNet.
- **Structure**: Frozen Base -> GlobalAveragePooling -> BN -> Dense(512) -> Dense(256) -> Output.
- **Training Strategy**:
  - **Phase 1**: Frozen base, trained head (lr=0.001) for 50 epochs.
  - **Phase 2**: Fine-tuned last 50 layers of base (lr=1e-4) for 80 epochs.

## Results & Analysis

### Performance Comparison

| Metric | Custom CNN | EfficientNetB0 |
| :--- | :--- | :--- |
| **Test Accuracy** | **93.80%** | **97.76%** |
| **Best Val Accuracy** | ~39% (Initial) -> Improved | **94.7%+** |

### Class-wise Performance (F1-Score)

| Class | CNN F1 | EfficientNetB0 F1 | Improvement |
| :--- | :--- | :--- | :--- |
| **Calculus** | 0.94 | **0.97** | +0.03 |
| **Caries** | 0.68 | **0.81** | +0.13 |
| **Hypodontia** | 0.96 | **0.99** | +0.03 |
| **Mouth Ulcer** | 0.85 | **0.96** | +0.11 |
| **Discoloration**| 0.96 | **0.99** | +0.03 |

### Key Findings
1.  **Superiority of Transfer Learning**: EfficientNetB0 achieved near-perfect classification for 'Hypodontia' and 'Tooth Discoloration' (99% F1).
2.  **Handling Difficult Classes**: 'Caries' was the hardest class to classify. The Custom CNN struggled significantly (68% F1), while EfficientNetB0 improved this to 81%, likely due to learned feature extractors from ImageNet being more robust.
3.  **Stability**: EfficientNetB0 showed faster convergence and higher stability during training compared to the custom CNN.

## Conclusion
The **EfficientNetB0** model is the recommended solution for this dental classification task. It provides a **3.96% absolute improvement** in accuracy over the custom CNN and, crucially, offers much better reliability in detecting 'Caries', the most challenging category in the dataset.

Conclusion
The EfficientNetB0 model is the recommended solution for this dental classification task. It provides a 3.96% absolute improvement in accuracy over the custom CNN and, crucially, offers much better reliability in detecting 'Caries', the most challenging category in the dataset.




