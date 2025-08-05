# Handwritten Text Recognition using EMNIST Datasets

<img width="1237" height="685" alt="image" src="https://github.com/user-attachments/assets/86a80604-40da-49e8-8f29-97a4bb20d253" />

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

* Uses the trained model to classify drawn characters

  
<img width="1238" height="745" alt="image" src="https://github.com/user-attachments/assets/8628efcc-89df-483e-85c9-46a8bf45f473" />


---

## ✨ Features

- **Real-time Drawing**: Draw characters directly on the canvas.
- **Image Loading**: Load handwritten images from your files.
- **Character Segmentation**: Automatically detects and separates individual characters from a word or sentence.
- **Confidence Scoring**: Displays model prediction confidence for each recognized character.
- **Visual Feedback**: Shows segmented character images alongside predictions for better understanding.
- **Custom Label Mapping**: Supports digits (0-9), uppercase letters (A-Z), and select lowercase letters (a, b, d, e, f, g, h, n, q, r, t).

---

## 🚀 Quick Start

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

## ✨ Features

* **Real-time Drawing**: Draw characters directly on the canvas.
* **Image Loading**: Load handwritten images from your files.
* **Character Segmentation**: Automatically detects and separates individual characters from a word or sentence.
* **Confidence Scoring**: Displays model prediction confidence for each recognized character.
* **Visual Feedback**: Shows segmented character images alongside predictions for better understanding.
* **Custom Label Mapping**: Supports digits (0-9), uppercase letters (A-Z), and select lowercase letters (a, b, d, e, f, g, h, n, q, r, t).

---

## 🚀 Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Place your trained EMNIST model**
   Put your `.h5` model file here:

   ```
   model/emnist_balanced_cnn_best.h5
   ```

3. **Run the application**

   ```bash
   python handwriting_app.py
   ```

4. **Start drawing or load an image**
   Use the canvas or load an image to recognize handwritten text!

---

## 🛠️ Tech Stack

* **GUI**: Tkinter (Python built-in)
* **ML Framework**: TensorFlow / Keras
* **Image Processing**: OpenCV, PIL
* **Model**: CNN trained on EMNIST Balanced dataset
* **Supported Characters**: `0-9`, `A-Z`, `a, b, d, e, f, g, h, n, q, r, t`

---

## 📊 Model Support

This canvas app is designed for CNN models trained on the **EMNIST Balanced** dataset.

* **Supported Character Classes**: 47
* **Character Types**: Digits, Uppercase Letters, and select Lowercase Letters
* Optimized for recognition in real-time use cases with clean segmentation and confidence visualization.

---

<img width="1255" height="721" alt="image" src="https://github.com/user-attachments/assets/0d59fe2c-1151-4825-80c3-a41cc3200177" />


---
