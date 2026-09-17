# Lab 01 — ML Model for Car Mileage Estimation

## 🎯 Objective

Build a **Linear Regression** model to predict car mileage (MPG — Miles Per Gallon) using the classic Auto MPG dataset. This lab introduces regression modeling, exploratory data analysis (EDA), and model evaluation metrics.

---

## 📊 Dataset

- **Name:** Auto MPG Dataset
- **Source:** Built-in via `seaborn` library (`seaborn.load_dataset('mpg')`)
- **Features:** Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, Origin
- **Target:** MPG (Miles Per Gallon)

No manual download required — the dataset loads automatically.

---

## 🔧 Dependencies

```
numpy, pandas, matplotlib, seaborn, scikit-learn
```

---

## ▶️ How to Run

```bash
cd Lab01_Car_Mileage_Estimation
python car_mileage_estimation.py
```

---

## 📤 Expected Output

1. **Dataset Info** — Shape, columns, first few rows, missing value summary
2. **EDA Plots** — Correlation heatmap, distribution plots, pairplots
3. **Model Training** — Linear Regression trained on 80/20 train-test split
4. **Evaluation Metrics** — MAE, RMSE, R² Score
5. **Prediction vs Actual** — Scatter plot comparing predicted and actual MPG values

---

## 📝 Key Concepts

- Exploratory Data Analysis (EDA)
- Linear Regression
- Train-Test Split
- Evaluation Metrics: MAE, RMSE, R²
- Data visualization with Matplotlib & Seaborn
