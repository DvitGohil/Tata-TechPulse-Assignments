"""
Lab 03 — Data Cleaning & Preprocessing Techniques
===================================================
Handle missing values, outliers, feature scaling, and encoding
on a simulated automotive dataset.

Key Concepts:
- Missing Value Imputation
- Outlier Detection (IQR)
- Feature Scaling (StandardScaler, MinMaxScaler)
- Categorical Encoding (One-Hot, Label)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ============================================================
# 1. Generate Synthetic Automotive Dataset
# ============================================================
print("=" * 60)
print("LAB 03 — Data Cleaning & Preprocessing Techniques")
print("=" * 60)

np.random.seed(42)
n_samples = 200

data = {
    'Mileage_kmpl': np.random.normal(15, 5, n_samples),
    'Engine_CC': np.random.normal(1500, 500, n_samples),
    'Horsepower': np.random.normal(120, 40, n_samples),
    'Weight_kg': np.random.normal(1200, 300, n_samples),
    'Age_years': np.random.randint(0, 20, n_samples).astype(float),
    'Fuel_Type': np.random.choice(['Petrol', 'Diesel', 'CNG', 'Electric'], n_samples),
    'Transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
    'Price_Lakhs': np.random.normal(10, 5, n_samples),
}

df = pd.DataFrame(data)

# Inject missing values
missing_indices = np.random.choice(n_samples, 20, replace=False)
df.loc[missing_indices[:7], 'Mileage_kmpl'] = np.nan
df.loc[missing_indices[7:12], 'Horsepower'] = np.nan
df.loc[missing_indices[12:17], 'Engine_CC'] = np.nan
df.loc[missing_indices[17:], 'Age_years'] = np.nan

# Inject outliers
df.loc[0, 'Price_Lakhs'] = 80.0
df.loc[1, 'Mileage_kmpl'] = 55.0
df.loc[2, 'Horsepower'] = 400.0
df.loc[3, 'Weight_kg'] = 3500.0

print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"\n[LOOK] First 5 Rows:")
print(df.head())

# ============================================================
# 2. Missing Value Analysis
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Missing Value Analysis")
print("-" * 40)

missing_summary = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing_summary, 'Missing %': missing_pct})
print(f"\n[X] Missing Values:\n{missing_df[missing_df['Missing Count'] > 0]}")

# Visualize missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd')
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.savefig('missing_values_heatmap.png', dpi=150)
plt.show()
print("[OK] Missing values heatmap saved")

# ============================================================
# 3. Handle Missing Values
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Handling Missing Values")
print("-" * 40)

# Impute numerical columns with median
num_cols = ['Mileage_kmpl', 'Engine_CC', 'Horsepower', 'Age_years']
for col in num_cols:
    median_val = df[col].median()
    count_before = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f"  {col}: Filled {count_before} missing values with median = {median_val:.2f}")

print(f"\n[OK] Missing values after imputation: {df.isnull().sum().sum()}")

# ============================================================
# 4. Outlier Detection & Treatment
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Outlier Detection (IQR Method)")
print("-" * 40)

numeric_cols = ['Mileage_kmpl', 'Engine_CC', 'Horsepower', 'Weight_kg', 'Price_Lakhs']

# Box plots before outlier removal
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(18, 5))
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col, fontsize=10)
plt.suptitle('Box Plots — Before Outlier Treatment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots_before.png', dpi=150)
plt.show()

# IQR-based outlier removal
df_clean = df.copy()
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    print(f"  {col}: {outliers} outliers detected (IQR: [{lower:.2f}, {upper:.2f}])")

print(f"\n[CHART] Shape after outlier removal: {df_clean.shape} (was {df.shape})")

# Box plots after outlier removal
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(18, 5))
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df_clean[col].dropna())
    axes[i].set_title(col, fontsize=10)
plt.suptitle('Box Plots — After Outlier Treatment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots_after.png', dpi=150)
plt.show()
print("[OK] Box plot comparisons saved")

# ============================================================
# 5. Feature Scaling
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Feature Scaling")
print("-" * 40)

scale_cols = ['Mileage_kmpl', 'Engine_CC', 'Horsepower', 'Weight_kg', 'Price_Lakhs']

# StandardScaler (Z-score normalization)
scaler_standard = StandardScaler()
df_standard = df_clean.copy()
df_standard[scale_cols] = scaler_standard.fit_transform(df_clean[scale_cols])

# MinMaxScaler (0-1 normalization)
scaler_minmax = MinMaxScaler()
df_minmax = df_clean.copy()
df_minmax[scale_cols] = scaler_minmax.fit_transform(df_clean[scale_cols])

print("\n[CHART] StandardScaler (Z-score) — First 3 rows:")
print(df_standard[scale_cols].head(3))
print("\n[CHART] MinMaxScaler (0-1) — First 3 rows:")
print(df_minmax[scale_cols].head(3))

# Comparison visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].set_title('Original', fontsize=12, fontweight='bold')
df_clean[scale_cols].hist(ax=axes[0], bins=20, color='steelblue', alpha=0.7)

axes[1].set_title('StandardScaler', fontsize=12, fontweight='bold')
df_standard[scale_cols].hist(ax=axes[1], bins=20, color='coral', alpha=0.7)

axes[2].set_title('MinMaxScaler', fontsize=12, fontweight='bold')
df_minmax[scale_cols].hist(ax=axes[2], bins=20, color='seagreen', alpha=0.7)

plt.suptitle('Feature Scaling Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_scaling_comparison.png', dpi=150)
plt.show()
print("[OK] Scaling comparison saved")

# ============================================================
# 6. Categorical Encoding
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: Categorical Encoding")
print("-" * 40)

# Label Encoding for Transmission
le = LabelEncoder()
df_clean['Transmission_Encoded'] = le.fit_transform(df_clean['Transmission'])
print(f"\n[TAG]  Label Encoding (Transmission): {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding for Fuel_Type
df_encoded = pd.get_dummies(df_clean, columns=['Fuel_Type'], prefix='Fuel')
print(f"\n[FIRE] One-Hot Encoding (Fuel_Type) — New columns: {[c for c in df_encoded.columns if 'Fuel_' in c]}")
print(f"\n[CHART] Final Encoded Dataset Shape: {df_encoded.shape}")
print(df_encoded.head())

print("\n" + "=" * 60)
print("[OK] Lab 03 Complete!")
print("=" * 60)
