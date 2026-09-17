# 🤖 Tata Tech Pulse— Lab Assignments

> **Course:** Applied AI ML | **Track:** AI & ML | **Level:** Intermediate  
> **Program:** TechPulse FY-26 — Tata Technologies Ltd.

---

## 📋 Overview

This repository contains **10 hands-on lab assignments** covering the full spectrum of Applied AI & Machine Learning — from classical ML and data preprocessing to deep learning, computer vision, NLP, and MLOps.

Each lab is self-contained in its own folder with:
- ✅ Python source code (`.py`)
- ✅ Individual `README.md` with objectives, instructions, and expected output
- ✅ Synthetic / built-in datasets (no manual downloads needed for most labs)

---

## 📂 Repository Structure

```
assignments/
├── README.md                              ← You are here
├── requirements.txt                       ← All Python dependencies
│
├── Lab01_Car_Mileage_Estimation/          ← Linear Regression on Auto MPG
├── Lab02_Simulated_Driving_Agent/         ← Genetic Algorithm driving agent
├── Lab03_Data_Cleaning_Preprocessing/     ← Data wrangling & EDA
├── Lab04_Vehicle_Price_Prediction/        ← Decision Tree & Random Forest
├── Lab05_Predictive_Maintenance/          ← Sensor-based failure classification
├── Lab06_Traffic_Sign_CNN/                ← CNN on GTSRB dataset
├── Lab07_Pedestrian_Detection/            ← HOG + SVM with OpenCV
├── Lab08_Sentiment_Analysis_LSTM/         ← LSTM sentiment classifier
├── Lab09_Feature_Importance/              ← Feature importance visualization
└── Lab10_MLOps_Workflow/                  ← MLflow + Docker pipeline
```

---

## 🧪 Lab Assignments

| # | Lab Title | Key Concepts | Folder |
|:-:|-----------|-------------|--------|
| 1 | **Car Mileage Estimation** | Linear Regression, EDA, Evaluation Metrics | [`Lab01`](./Lab01_Car_Mileage_Estimation/) |
| 2 | **Simulated Driving Agent** | Genetic Algorithms, Agent-based Simulation | [`Lab02`](./Lab02_Simulated_Driving_Agent/) |
| 3 | **Data Cleaning & Preprocessing** | Missing Values, Outliers, Scaling, Encoding | [`Lab03`](./Lab03_Data_Cleaning_Preprocessing/) |
| 4 | **Vehicle Price Prediction** | Decision Trees, Random Forest, Feature Engineering | [`Lab04`](./Lab04_Vehicle_Price_Prediction/) |
| 5 | **Predictive Maintenance** | Classification, Sensor Data, Confusion Matrix | [`Lab05`](./Lab05_Predictive_Maintenance/) |
| 6 | **Traffic Sign Classification (CNN)** | CNNs, Image Preprocessing, GTSRB Dataset | [`Lab06`](./Lab06_Traffic_Sign_CNN/) |
| 7 | **Pedestrian Detection** | HOG Descriptors, SVM, OpenCV | [`Lab07`](./Lab07_Pedestrian_Detection/) |
| 8 | **Sentiment Analysis (LSTM)** | RNN/LSTM, Text Preprocessing, NLP | [`Lab08`](./Lab08_Sentiment_Analysis_LSTM/) |
| 9 | **Feature Importance Visualization** | Permutation Importance, Random Forest, XGBoost | [`Lab09`](./Lab09_Feature_Importance/) |
| 10 | **MLOps Workflow Simulation** | MLflow, Docker, CI/CD Pipeline | [`Lab10`](./Lab10_MLOps_Workflow/) |

---

## 🛠️ Prerequisites

- Python 3.8 or higher
- Basic programming skills in Python
- Familiarity with data structures and algorithms
- Introductory understanding of ML concepts
- Comfort with using terminal / command line

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/DvitGohil/Applied-AI-ML-Labs.git
cd Applied-AI-ML-Labs
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Each lab can be run independently. Navigate to the lab folder and execute the Python script:

```bash
# Example: Run Lab 1
cd Lab01_Car_Mileage_Estimation
python car_mileage_estimation.py

# Example: Run Lab 5
cd Lab05_Predictive_Maintenance
python predictive_maintenance.py
```

> **Note:** Some labs (Lab 6, Lab 10) may require additional setup — refer to the individual lab README for details.

---

## 📚 Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.8+ |
| NumPy | ≥ 1.24.0 |
| Pandas | ≥ 2.0.0 |
| Scikit-learn | ≥ 1.3.0 |
| Matplotlib | ≥ 3.7.0 |
| Seaborn | ≥ 0.12.0 |
| TensorFlow / Keras | ≥ 2.13.0 |
| OpenCV | ≥ 4.8.0 |
| MLflow | ≥ 2.5.0 |
| Docker | Latest |

---

## 📖 Recommended Books

- *Introduction to AI & Machine Learning* — Munesh Chandra Trivedi & Ankit Srivastava
- *Python Machine Learning* — Sebastian Raschka & Vahid Mirjalili
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — Aurélien Géron

---

## 📝 License

This project is for educational purposes as part of the TechPulse FY-26 program by Tata Technologies Ltd.

---

## 🙋 Author

**Dvit Gohil**  
MIT WPU | Applied AI & ML Track
