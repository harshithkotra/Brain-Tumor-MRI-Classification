# 🧠 Brain Tumor MRI Image Classification

An AI-assisted medical imaging application that classifies brain MRI images into four categories using Deep Learning and Transfer Learning. The project compares a custom CNN model with an EfficientNetB0-based transfer learning model and deploys the best-performing model through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Brain tumor classification from MRI images is an important medical imaging problem where different tumor types can have visually similar characteristics.

This project develops a deep learning-based classification system capable of identifying MRI images as:

- Glioma
- Meningioma
- No Tumor
- Pituitary

A custom CNN was developed as a baseline model, followed by transfer learning using EfficientNetB0 with ImageNet pretrained weights.

The final EfficientNetB0 model was integrated into a professional Streamlit application for real-time MRI image classification.

> ⚠️ This application is intended for educational and research purposes and should be considered an AI-assisted decision-support tool, not a standalone medical diagnostic system.

---

## 🎯 Problem Statement

The objective of this project is to develop a deep learning-based solution for multi-class brain MRI image classification.

The system aims to:

1. Preprocess and standardize MRI images.
2. Apply image augmentation to improve model generalization.
3. Build a custom CNN model from scratch.
4. Apply transfer learning using EfficientNetB0.
5. Evaluate models using multiple performance metrics.
6. Compare custom CNN and transfer learning performance.
7. Deploy the trained model using Streamlit.
8. Provide real-time predictions with confidence scores.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Streamlit
- EfficientNetB0
- Transfer Learning
- Image Classification

---

## 🧠 Machine Learning Approach

### 1. Data Preprocessing

The MRI images were:

- Resized to `224 × 224` pixels.
- Prepared according to the requirements of the selected model.
- Organized into four target classes.

### 2. Data Augmentation

Training images were augmented using:

- Rotation
- Width and height shifting
- Zoom
- Horizontal flipping
- Brightness variation

These transformations help the model learn more robust visual patterns.

---

## 🏗️ Model Development

### Custom CNN

A convolutional neural network was developed from scratch using:

- Convolutional layers
- Batch Normalization
- Max Pooling
- Dense layers
- Dropout
- Softmax classification layer

The custom CNN was used as the baseline model.

### EfficientNetB0 Transfer Learning

EfficientNetB0 pretrained on ImageNet was selected as the main transfer learning architecture.

The original classification head was replaced with a custom classification head designed for the four MRI classes.

The pretrained base layers were initially frozen while the new classification layers were trained.

---

## 📊 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix
- Training and validation accuracy
- Training and validation loss

---

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Custom CNN | 21.62% | 31.30% | 21.62% | 11.88% |
| EfficientNetB0 | **59.46%** | **61.82%** | **59.46%** | **59.62%** |

### Key Result

EfficientNetB0 achieved a **37.84 percentage-point improvement in accuracy** compared with the custom CNN baseline.

This demonstrates the benefit of using transfer learning for the available MRI classification dataset.

---

## 📂 Dataset

The project uses a Brain Tumor MRI multi-class dataset containing four categories:

```text
glioma
meningioma
no_tumor
pituitary