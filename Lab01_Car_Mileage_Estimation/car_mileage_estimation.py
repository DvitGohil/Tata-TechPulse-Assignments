"""
Lab 01 — ML Model for Car Mileage Estimation
=============================================
Predict car mileage (MPG) using Linear Regression on the Auto MPG dataset.

Key Concepts:
- Exploratory Data Analysis (EDA)
- Linear Regression
- Model Evaluation (MAE, RMSE, R²)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 60)
print("LAB 01 — Car Mileage Estimation (Linear Regression)")
print("=" * 60)

df = sns.load_dataset('mpg')
print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"\n[INFO] Columns: {list(df.columns)}")
print(f"\n[LOOK] First 5 Rows:")
print(df.head())

# ============================================================
# 2. Data Cleaning
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Data Cleaning")
print("-" * 40)

print(f"\n[X] Missing Values:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\n[OK] After dropping nulls: {df.shape}")

# Drop the 'name' column (non-numeric, not useful for regression)
df = df.drop(columns=['name'])

# Convert 'origin' to categorical dummy variables
df = pd.get_dummies(df, columns=['origin'], drop_first=True)

print(f"\n[CHART] Final Dataset Shape: {df.shape}")
print(df.head())

# ============================================================
# 3. Exploratory Data Analysis (EDA)
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Exploratory Data Analysis")
print("-" * 40)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap — Auto MPG Dataset')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("[OK] Correlation heatmap saved as 'correlation_heatmap.png'")

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df['mpg'], kde=True, color='steelblue', bins=30)
plt.title('Distribution of MPG (Target Variable)')
plt.xlabel('Miles Per Gallon')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('mpg_distribution.png', dpi=150)
plt.show()
print("[OK] MPG distribution plot saved as 'mpg_distribution.png'")

# ============================================================
# 4. Feature Selection & Train-Test Split
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Feature Selection & Train-Test Split")
print("-" * 40)

X = df.drop(columns=['mpg'])
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# ============================================================
# 5. Model Training
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Model Training — Linear Regression")
print("-" * 40)

model = LinearRegression()
model.fit(X_train, y_train)

print("[OK] Model trained successfully!")
print(f"\n[MATH] Coefficients: {dict(zip(X.columns, np.round(model.coef_, 4)))}")
print(f"[MATH] Intercept: {model.intercept_:.4f}")

# ============================================================
# 6. Predictions & Evaluation
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: Predictions & Evaluation")
print("-" * 40)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n[CHART] Model Performance:")
print(f"   MAE  (Mean Absolute Error):  {mae:.4f}")
print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"   R²   (R-Squared Score):      {r2:.4f}")

# ============================================================
# 7. Prediction vs Actual Plot
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal', edgecolors='k', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Ideal Prediction')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
print("\n[OK] Actual vs Predicted plot saved as 'actual_vs_predicted.png'")

print("\n" + "=" * 60)
print("[OK] Lab 01 Complete!")
print("=" * 60)
