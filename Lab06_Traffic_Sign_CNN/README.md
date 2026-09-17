# Lab 06 — Traffic Sign Classification using CNN

## 🎯 Objective

Train a **Convolutional Neural Network (CNN)** to classify traffic signs using the **GTSRB** (German Traffic Sign Recognition Benchmark) dataset. This lab covers image preprocessing, CNN architecture design, and model evaluation for computer vision tasks.

---

## 📊 Dataset

- **Name:** GTSRB — German Traffic Sign Recognition Benchmark
- **Source:** Loaded via `tensorflow.keras.utils.get_file` (auto-download)
- **Classes:** 43 traffic sign categories
- **Images:** 32×32 RGB images

> **Note:** The dataset will be downloaded automatically on first run (~300 MB). Alternatively, you can use the built-in CIFAR-10 dataset as a fallback (the script supports both).

---

## 🔧 Dependencies

```
numpy, pandas, matplotlib, seaborn, tensorflow, keras, scikit-learn
```

---

## ▶️ How to Run

```bash
cd Lab06_Traffic_Sign_CNN
python traffic_sign_cnn.py
```

---

## 📤 Expected Output

1. **Dataset Overview** — Sample images from different classes
2. **CNN Architecture** — Model summary with layer details
3. **Training Progress** — Accuracy and loss curves over epochs
4. **Evaluation** — Test accuracy, confusion matrix
5. **Sample Predictions** — Grid of predicted vs actual labels

---

## 📝 Key Concepts

- Convolutional Neural Networks (Conv2D, MaxPooling, Flatten, Dense)
- Image Preprocessing & Data Augmentation
- Dropout for Regularization
- Training Curves (Accuracy, Loss)
- Confusion Matrix for Multi-class Classification
