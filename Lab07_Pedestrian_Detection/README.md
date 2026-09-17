# Lab 07 — Pedestrian Detection using OpenCV

## 🎯 Objective

Detect pedestrians in images using **OpenCV's HOG (Histogram of Oriented Gradients) + SVM** method. This lab demonstrates a classical computer vision approach to object detection used in automotive safety systems.

---

## 📊 Dataset

- **No external dataset required**
- The script generates synthetic test images with geometric shapes simulating pedestrian-like figures
- For real-world usage, you can replace with actual images from any pedestrian dataset

---

## 🔧 Dependencies

```
numpy, matplotlib, opencv-python
```

---

## ▶️ How to Run

```bash
cd Lab07_Pedestrian_Detection
python pedestrian_detection.py
```

---

## 📤 Expected Output

1. **HOG Feature Visualization** — HOG descriptor computed on sample images
2. **Detection Results** — Bounding boxes drawn around detected pedestrians
3. **Non-Maximum Suppression** — Overlapping detections merged
4. **Performance Summary** — Number of detections and confidence scores

---

## 📝 Key Concepts

- HOG (Histogram of Oriented Gradients) Descriptors
- SVM (Support Vector Machine) Classifier
- Sliding Window Detection
- Non-Maximum Suppression (NMS)
- OpenCV's built-in Pedestrian Detector
