# 🧠 Brain Tumor MRI Image Classification

A deep learning-based medical imaging project that classifies brain MRI images into four categories using a custom Convolutional Neural Network (CNN) and an EfficientNetB0 transfer learning model.

The project also includes a professional Streamlit web application that allows users to upload an MRI image and receive the predicted brain tumor category along with the prediction confidence.

---

## 📌 Project Overview

Brain tumor classification from MRI images is an important medical imaging problem where deep learning can assist in identifying different tumor categories from brain scans.

This project explores two approaches:

- Custom Convolutional Neural Network (CNN)
- EfficientNetB0 Transfer Learning

The models are trained and evaluated on a four-class brain MRI dataset. The best-performing model is integrated into a Streamlit application for interactive prediction.

> ⚠️ This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🎯 Problem Statement

The objective of this project is to develop a deep learning system capable of classifying brain MRI images into the following categories:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

The project focuses on comparing a custom CNN with a pretrained EfficientNetB0 model and deploying the selected model through a user-friendly web interface.

---

## 🧠 Classes

| Class | Description |
|---|---|
| Glioma | Brain tumor originating from glial cells |
| Meningioma | Tumor associated with the meninges |
| No Tumor | MRI image without a detected tumor category |
| Pituitary | Tumor associated with the pituitary gland |

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
- Git & GitHub

---

## 🔬 Machine Learning Approach

The project follows the following workflow:

```text
MRI Dataset
     ↓
Data Exploration
     ↓
Image Preprocessing
     ↓
Data Augmentation
     ↓
Custom CNN Training
     ↓
EfficientNetB0 Transfer Learning
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Streamlit Deployment
     ↓
MRI Image Prediction