# Lab 09 — Feature Importance Visualization

## 🎯 Objective

Visualize and interpret **feature importance** in automotive datasets using multiple methods — Random Forest importance, Permutation Importance, and correlation analysis. Understand which features drive model predictions.

---

## 📊 Dataset

- **Synthetic automotive dataset** generated programmatically
- Features: Engine_CC, Horsepower, Weight, Mileage, Age, Fuel_Type, Transmission, Safety_Rating, Brand_Reputation
- Target: Price (regression) and Quality_Class (classification)

No manual download required.

---

## 🔧 Dependencies

```
numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost
```

---

## ▶️ How to Run

```bash
cd Lab09_Feature_Importance
python feature_importance.py
```

---

## 📤 Expected Output

1. **Correlation Analysis** — Heatmap showing feature correlations
2. **Random Forest Importance** — Bar chart of Gini importance
3. **Permutation Importance** — Impact-based feature ranking
4. **XGBoost Importance** — Gradient boosting feature scores
5. **Comparison Dashboard** — Side-by-side comparison of all methods

---

## 📝 Key Concepts

- Feature Importance (Gini / MDI)
- Permutation Importance
- XGBoost Feature Importance
- Correlation Analysis
- Model Interpretability
