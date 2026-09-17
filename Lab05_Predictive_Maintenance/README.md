# Lab 05 — Predictive Maintenance from Sensor Logs

## 🎯 Objective

Build a **classification model** to predict component failures from simulated sensor data. This lab demonstrates how machine learning can be applied to **predictive maintenance** in automotive and industrial applications.

---

## 📊 Dataset

- **Synthetic sensor log dataset** generated programmatically
- Simulates real-time sensor readings from vehicle components
- Features: Temperature, Vibration, Pressure, RPM, Voltage, Operating_Hours
- Target: Failure (0 = Normal, 1 = Failure)

No manual download required.

---

## 🔧 Dependencies

```
numpy, pandas, matplotlib, seaborn, scikit-learn
```

---

## ▶️ How to Run

```bash
cd Lab05_Predictive_Maintenance
python predictive_maintenance.py
```

---

## 📤 Expected Output

1. **Dataset Overview** — Sensor readings and class distribution
2. **EDA** — Feature distributions by failure class
3. **Model Training** — Random Forest and Logistic Regression classifiers
4. **Confusion Matrix** — Visual confusion matrices for both models
5. **Classification Report** — Precision, Recall, F1-Score
6. **ROC Curve** — AUC comparison

---

## 📝 Key Concepts

- Binary Classification
- Sensor Data Analysis
- Confusion Matrix & Classification Report
- ROC-AUC Curve
- Predictive Maintenance in Industry
