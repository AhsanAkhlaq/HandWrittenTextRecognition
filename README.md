# Handwritten Text Recognition using EMNIST Datasets

This project demonstrates handwritten text recognition using deep learning on two different datasets from the EMNIST (Extended MNIST) dataset collection:

* **EMNIST ByClass** (62 classes: digits + uppercase + lowercase letters)
* **EMNIST Balanced** (47 classes: balanced mix of digits and letters)

Both models share the **same CNN-based architecture** and are evaluated on accuracy, loss, precision, recall, F1-score, and ROC-AUC metrics.

---

## 🧠 Model Architecture

* Deep Convolutional Neural Network (CNN)
* Input size: 28x28 grayscale images
* Optimizer: Adam
* Loss function: Sparse Categorical Crossentropy
* Metrics: Accuracy, Top-5 Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📊 Dataset Overview

### EMNIST ByClass

* Training Samples: **697,931**
* Test Samples: **116,322**
* Total Samples: **814,253**
* Unique Classes: **62** (includes digits, uppercase & lowercase letters)

### EMNIST Balanced

* Training Samples: **112,799**
* Test Samples: **18,799**
* Total Samples: **131,598**
* Unique Classes: **47** (balanced distribution)

---

## 🧪 Model Evaluation

### EMNIST ByClass Model

* **Test Accuracy**: 87.38%
* **Test Loss**: 0.3392
* **Top-5 Accuracy**: 99.75%
* **Precision (macro)**: 0.7781
* **Recall (macro)**: 0.7402
* **F1-Score (macro)**: 0.7301
* **ROC-AUC (macro)**: 0.9965
* **Epochs**: 30
* **Model Parameters**: 502,686

### EMNIST Balanced Model

* **Test Accuracy**: 88.52%
* **Test Loss**: 0.3216
* **Top-5 Accuracy**: 99.52%
* **Precision (macro)**: 0.8951
* **Recall (macro)**: 0.8855
* **F1-Score (macro)**: 0.8828
* **ROC-AUC (macro)**: 0.9972
* **Epochs**: 20
* **Model Parameters**: 498,831

---

## 🆚 Comparison & Recommendation

| Metric            | EMNIST ByClass  | EMNIST Balanced |
| ----------------- | --------------- | --------------- |
| Accuracy          | 87.38%          | **88.52%** ✅    |
| Top-5 Accuracy    | **99.75%** ✅    | 99.52%          |
| Precision (macro) | 0.7781          | **0.8951** ✅    |
| Recall (macro)    | 0.7402          | **0.8855** ✅    |
| F1-Score (macro)  | 0.7301          | **0.8828** ✅    |
| ROC-AUC (macro)   | 0.9965          | **0.9972** ✅    |
| Training Time     | 30 epochs       | **20 epochs** ✅ |
| Classes           | **62** (More) ✅ | 47              |

### Verdict:

* Use **EMNIST ByClass** if you need **full coverage of characters (digits + upper/lowercase letters)**.
* Use **EMNIST Balanced** if you want **higher accuracy and better performance** on a **more focused dataset**.

---

## 🖊️ Canvas App

A custom Python-based **Tkinter Canvas GUI** is included, allowing you to **draw characters or words**, and the model will predict the handwritten text in real-time using the trained model.

* Preprocessing: Image mirroring and rotation for correct orientation
* Uses the trained model to classify drawn characters

---
