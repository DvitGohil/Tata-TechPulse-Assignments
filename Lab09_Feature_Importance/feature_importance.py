"""
Lab 09 — Feature Importance Visualization
===========================================
Visualize and interpret feature importance in automotive datasets
using Random Forest, Permutation Importance, and XGBoost.

Key Concepts:
- Feature Importance (Gini / MDI)
- Permutation Importance
- XGBoost Feature Importance
- Correlation Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ============================================================
# 1. Generate Synthetic Automotive Dataset
# ============================================================
print("=" * 60)
print("LAB 09 — Feature Importance Visualization")
print("=" * 60)

np.random.seed(42)
n = 500

data = {
    'Engine_CC': np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], n),
    'Horsepower': np.random.randint(60, 250, n),
    'Weight_kg': np.random.normal(1300, 300, n).astype(int),
    'Mileage_kmpl': np.random.normal(15, 5, n),
    'Age_years': np.random.randint(0, 15, n),
    'Fuel_Type': np.random.choice(['Petrol', 'Diesel', 'CNG', 'Electric'], n),
    'Transmission': np.random.choice(['Manual', 'Automatic'], n),
    'Safety_Rating': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.25, 0.35, 0.25]),
    'Brand_Reputation': np.random.randint(1, 11, n),
}

# Generate price with known feature relationships
price = (
    0.005 * np.array(data['Engine_CC'])
    + 0.04 * np.array(data['Horsepower'])
    - 0.001 * np.array(data['Weight_kg'])
    + 0.1 * np.array(data['Mileage_kmpl'])
    - 0.8 * np.array(data['Age_years'])
    + 1.5 * np.array(data['Safety_Rating'])
    + 0.5 * np.array(data['Brand_Reputation'])
    + np.random.normal(0, 2, n)
    + 8
)
data['Price_Lakhs'] = np.clip(price, 1, 50).round(2)

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

# Encode categoricals
le_fuel = LabelEncoder()
le_trans = LabelEncoder()
df['Fuel_Encoded'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Trans_Encoded'] = le_trans.fit_transform(df['Transmission'])

feature_cols = ['Engine_CC', 'Horsepower', 'Weight_kg', 'Mileage_kmpl',
                'Age_years', 'Fuel_Encoded', 'Trans_Encoded',
                'Safety_Rating', 'Brand_Reputation']
feature_names = ['Engine', 'HP', 'Weight', 'Mileage', 'Age',
                 'Fuel', 'Trans', 'Safety', 'Reputation']

X = df[feature_cols]
y = df['Price_Lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# ============================================================
# 3. Correlation Analysis
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Correlation Analysis")
print("-" * 40)

plt.figure(figsize=(10, 8))
corr_matrix = df[feature_cols + ['Price_Lakhs']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()
print("[OK] Correlation matrix saved")

# Print top correlations with target
target_corr = corr_matrix['Price_Lakhs'].drop('Price_Lakhs').sort_values(ascending=False)
print(f"\n[CHART] Correlation with Price:")
for feat, corr in target_corr.items():
    bar = '#' * int(abs(corr) * 20)
    sign = '+' if corr > 0 else '-'
    print(f"  {feat:20s} {sign}{abs(corr):.3f} {bar}")

# ============================================================
# 4. Random Forest Feature Importance
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Random Forest Feature Importance (Gini / MDI)")
print("-" * 40)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)

rf_importance = rf_model.feature_importances_
rf_sorted_idx = np.argsort(rf_importance)

print(f"\n[TREE] Random Forest R²: {rf_score:.4f}")
print(f"\n[CHART] Feature Importance (Gini):")
for idx in rf_sorted_idx[::-1]:
    bar = '#' * int(rf_importance[idx] * 50)
    print(f"  {feature_names[idx]:12s} {rf_importance[idx]:.4f} {bar}")

# ============================================================
# 5. Permutation Importance
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Permutation Importance")
print("-" * 40)

perm_result = permutation_importance(rf_model, X_test, y_test,
                                      n_repeats=10, random_state=42)

perm_importance = perm_result.importances_mean
perm_sorted_idx = np.argsort(perm_importance)

print(f"\n[CHART] Permutation Importance:")
for idx in perm_sorted_idx[::-1]:
    bar = '#' * int(perm_importance[idx] * 50)
    print(f"  {feature_names[idx]:12s} {perm_importance[idx]:.4f} ± {perm_result.importances_std[idx]:.4f} {bar}")

# ============================================================
# 6. XGBoost Feature Importance
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: XGBoost Feature Importance")
print("-" * 40)

xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_score = xgb_model.score(X_test, y_test)

xgb_importance = xgb_model.feature_importances_
xgb_sorted_idx = np.argsort(xgb_importance)

print(f"\n[ROCKET] XGBoost R²: {xgb_score:.4f}")
print(f"\n[CHART] XGBoost Feature Importance:")
for idx in xgb_sorted_idx[::-1]:
    bar = '#' * int(xgb_importance[idx] * 50)
    print(f"  {feature_names[idx]:12s} {xgb_importance[idx]:.4f} {bar}")

# ============================================================
# 7. Comparison Dashboard
# ============================================================
print("\n" + "-" * 40)
print("STEP 7: Comparison Dashboard")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Random Forest Importance
ax = axes[0][0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rf_sorted_idx)))
ax.barh(range(len(rf_sorted_idx)), rf_importance[rf_sorted_idx], color=colors)
ax.set_yticks(range(len(rf_sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in rf_sorted_idx])
ax.set_xlabel('Importance')
ax.set_title(f'Random Forest (Gini) — R²={rf_score:.3f}', fontweight='bold')

# Plot 2: Permutation Importance
ax = axes[0][1]
ax.barh(range(len(perm_sorted_idx)), perm_importance[perm_sorted_idx],
        xerr=perm_result.importances_std[perm_sorted_idx],
        color='coral', capsize=3)
ax.set_yticks(range(len(perm_sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in perm_sorted_idx])
ax.set_xlabel('Mean Importance')
ax.set_title('Permutation Importance', fontweight='bold')

# Plot 3: XGBoost Importance
ax = axes[1][0]
colors_xgb = plt.cm.plasma(np.linspace(0.2, 0.8, len(xgb_sorted_idx)))
ax.barh(range(len(xgb_sorted_idx)), xgb_importance[xgb_sorted_idx], color=colors_xgb)
ax.set_yticks(range(len(xgb_sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in xgb_sorted_idx])
ax.set_xlabel('Importance')
ax.set_title(f'XGBoost — R²={xgb_score:.3f}', fontweight='bold')

# Plot 4: Combined Ranking
ax = axes[1][1]
comparison_df = pd.DataFrame({
    'Feature': feature_names,
    'RF_Rank': len(feature_names) - np.argsort(np.argsort(rf_importance)),
    'Perm_Rank': len(feature_names) - np.argsort(np.argsort(perm_importance)),
    'XGB_Rank': len(feature_names) - np.argsort(np.argsort(xgb_importance)),
})
comparison_df['Avg_Rank'] = comparison_df[['RF_Rank', 'Perm_Rank', 'XGB_Rank']].mean(axis=1)
comparison_df = comparison_df.sort_values('Avg_Rank')

x = np.arange(len(comparison_df))
width = 0.25
ax.bar(x - width, comparison_df['RF_Rank'], width, label='Random Forest', color='steelblue')
ax.bar(x, comparison_df['Perm_Rank'], width, label='Permutation', color='coral')
ax.bar(x + width, comparison_df['XGB_Rank'], width, label='XGBoost', color='seagreen')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
ax.set_ylabel('Rank (lower = more important)')
ax.set_title('Feature Ranking Comparison', fontweight='bold')
ax.legend()

plt.suptitle('Feature Importance Analysis — Dashboard', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_dashboard.png', dpi=150)
plt.show()
print("[OK] Dashboard saved as 'feature_importance_dashboard.png'")

# Print final ranking
print(f"\n[BEST] Overall Feature Ranking (avg across methods):")
for i, row in comparison_df.iterrows():
    print(f"  {row['Avg_Rank']:.1f}  {row['Feature']}")

print("\n" + "=" * 60)
print("[OK] Lab 09 Complete!")
print("=" * 60)
