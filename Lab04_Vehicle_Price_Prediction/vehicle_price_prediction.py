"""
Lab 04 — Vehicle Price Prediction
==================================
Use Decision Tree and Random Forest regression to predict vehicle prices.

Key Concepts:
- Decision Tree Regression
- Random Forest Ensemble
- Feature Engineering & Importance
- Model Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. Generate Synthetic Vehicle Dataset
# ============================================================
print("=" * 60)
print("LAB 04 — Vehicle Price Prediction")
print("=" * 60)

np.random.seed(42)
n = 500

brands = ['Maruti', 'Hyundai', 'Tata', 'Honda', 'Toyota', 'Kia', 'Mahindra']
fuel_types = ['Petrol', 'Diesel', 'CNG']
transmissions = ['Manual', 'Automatic']

data = {
    'Brand': np.random.choice(brands, n),
    'Age_years': np.random.randint(0, 15, n),
    'Mileage_km': np.random.randint(5000, 200000, n),
    'Engine_CC': np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2500], n),
    'Horsepower': np.random.randint(60, 200, n),
    'Fuel_Type': np.random.choice(fuel_types, n),
    'Transmission': np.random.choice(transmissions, n),
    'Owner_Count': np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.1]),
}

# Generate price based on features (with some noise)
base_price = (
    0.005 * np.array(data['Engine_CC'])
    + 0.03 * np.array(data['Horsepower'])
    - 0.5 * np.array(data['Age_years'])
    - 0.00003 * np.array(data['Mileage_km'])
    - 1.0 * np.array(data['Owner_Count'])
    + np.random.normal(0, 1.5, n)
    + 10
)
data['Price_Lakhs'] = np.clip(base_price, 1.0, 50.0)

df = pd.DataFrame(data)

print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"\n[LOOK] First 5 Rows:")
print(df.head())
print(f"\n[CHART] Statistics:")
print(df.describe().round(2))

# ============================================================
# 2. Data Preprocessing
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Data Preprocessing")
print("-" * 40)

# Label encode categorical columns
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
le_trans = LabelEncoder()

df['Brand_Encoded'] = le_brand.fit_transform(df['Brand'])
df['Fuel_Encoded'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Trans_Encoded'] = le_trans.fit_transform(df['Transmission'])

# Feature matrix and target
feature_cols = ['Brand_Encoded', 'Age_years', 'Mileage_km', 'Engine_CC',
                'Horsepower', 'Fuel_Encoded', 'Trans_Encoded', 'Owner_Count']

X = df[feature_cols]
y = df['Price_Lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# ============================================================
# 3. Decision Tree Regressor
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Decision Tree Regressor")
print("-" * 40)

dt_model = DecisionTreeRegressor(max_depth=8, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print(f"\n[TREE] Decision Tree Performance:")
print(f"   MAE:  {dt_mae:.4f}")
print(f"   RMSE: {dt_rmse:.4f}")
print(f"   R²:   {dt_r2:.4f}")

# ============================================================
# 4. Random Forest Regressor
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Random Forest Regressor")
print("-" * 40)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"\n[TREE] Random Forest Performance:")
print(f"   MAE:  {rf_mae:.4f}")
print(f"   RMSE: {rf_rmse:.4f}")
print(f"   R²:   {rf_r2:.4f}")

# ============================================================
# 5. Model Comparison
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Model Comparison")
print("-" * 40)

comparison = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Decision Tree': [dt_mae, dt_rmse, dt_r2],
    'Random Forest': [rf_mae, rf_rmse, rf_r2]
})
print(f"\n{comparison.to_string(index=False)}")

# ============================================================
# 6. Visualizations
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Actual vs Predicted — Decision Tree
axes[0].scatter(y_test, dt_pred, alpha=0.6, color='coral', edgecolors='k', s=40)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=2)
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'Decision Tree (R²={dt_r2:.3f})', fontweight='bold')

# Plot 2: Actual vs Predicted — Random Forest
axes[1].scatter(y_test, rf_pred, alpha=0.6, color='teal', edgecolors='k', s=40)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=2)
axes[1].set_xlabel('Actual Price')
axes[1].set_ylabel('Predicted Price')
axes[1].set_title(f'Random Forest (R²={rf_r2:.3f})', fontweight='bold')

# Plot 3: Feature Importance
feature_names = ['Brand', 'Age', 'Mileage', 'Engine', 'HP', 'Fuel', 'Trans', 'Owners']
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)
axes[2].barh(range(len(sorted_idx)), importances[sorted_idx], color='steelblue')
axes[2].set_yticks(range(len(sorted_idx)))
axes[2].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[2].set_xlabel('Importance')
axes[2].set_title('Feature Importance (Random Forest)', fontweight='bold')

plt.suptitle('Vehicle Price Prediction — Model Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('price_prediction_results.png', dpi=150)
plt.show()
print("[OK] Results saved as 'price_prediction_results.png'")

print("\n" + "=" * 60)
print("[OK] Lab 04 Complete!")
print("=" * 60)
