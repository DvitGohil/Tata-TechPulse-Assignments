"""
Lab 05 — Predictive Maintenance from Sensor Logs
=================================================
Classify component failures from simulated sensor data using ML models.

Key Concepts:
- Binary Classification
- Sensor Data Analysis
- Confusion Matrix & Classification Report
- ROC-AUC Curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score)

# ============================================================
# 1. Generate Synthetic Sensor Dataset
# ============================================================
print("=" * 60)
print("LAB 05 — Predictive Maintenance from Sensor Logs")
print("=" * 60)

np.random.seed(42)
n = 1000

# Normal operating conditions
normal_data = {
    'Temperature_C': np.random.normal(75, 10, n),
    'Vibration_mm_s': np.random.normal(2.5, 0.8, n),
    'Pressure_bar': np.random.normal(5.0, 1.0, n),
    'RPM': np.random.normal(3000, 500, n),
    'Voltage_V': np.random.normal(12.0, 0.5, n),
    'Operating_Hours': np.random.randint(100, 5000, n).astype(float),
}

# Create failure condition: higher temp, vibration, lower voltage
n_fail = 200
failure_indices = np.random.choice(n, n_fail, replace=False)

# Modify sensor values for failure cases
for idx in failure_indices:
    normal_data['Temperature_C'][idx] += np.random.normal(30, 10)
    normal_data['Vibration_mm_s'][idx] += np.random.normal(3, 1)
    normal_data['Pressure_bar'][idx] += np.random.normal(2, 0.5)
    normal_data['Voltage_V'][idx] -= np.random.normal(1.5, 0.5)

# Create failure labels
labels = np.zeros(n)
labels[failure_indices] = 1

normal_data['Failure'] = labels
df = pd.DataFrame(normal_data)

print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"\n[CHART] Class Distribution:")
print(df['Failure'].value_counts().rename({0: 'Normal', 1: 'Failure'}))
print(f"\n[LOOK] First 5 Rows:")
print(df.head())

# ============================================================
# 2. Exploratory Data Analysis
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Exploratory Data Analysis")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sensor_cols = ['Temperature_C', 'Vibration_mm_s', 'Pressure_bar',
               'RPM', 'Voltage_V', 'Operating_Hours']

for i, col in enumerate(sensor_cols):
    ax = axes[i // 3][i % 3]
    for label, color in [(0, 'steelblue'), (1, 'red')]:
        subset = df[df['Failure'] == label]
        ax.hist(subset[col], bins=30, alpha=0.6, color=color,
                label='Normal' if label == 0 else 'Failure')
    ax.set_title(col, fontweight='bold')
    ax.legend()

plt.suptitle('Sensor Reading Distributions by Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sensor_distributions.png', dpi=150)
plt.show()
print("[OK] Sensor distribution plots saved")

# ============================================================
# 3. Data Preparation
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Data Preparation")
print("-" * 40)

X = df.drop(columns=['Failure'])
y = df['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# ============================================================
# 4. Model Training
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Model Training")
print("-" * 40)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n[TREE] Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(classification_report(y_test, rf_pred, target_names=['Normal', 'Failure']))

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"[UP] Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred, target_names=['Normal', 'Failure']))

# ============================================================
# 5. Visualizations
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Confusion Matrix — Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Random Forest — Confusion Matrix', fontweight='bold')

# Confusion Matrix — Logistic Regression
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Logistic Regression — Confusion Matrix', fontweight='bold')

# ROC Curve
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_auc = auc(rf_fpr, rf_tpr)
lr_auc = auc(lr_fpr, lr_tpr)

axes[2].plot(rf_fpr, rf_tpr, color='steelblue', linewidth=2,
             label=f'Random Forest (AUC={rf_auc:.3f})')
axes[2].plot(lr_fpr, lr_tpr, color='coral', linewidth=2,
             label=f'Logistic Regression (AUC={lr_auc:.3f})')
axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve Comparison', fontweight='bold')
axes[2].legend()

plt.suptitle('Predictive Maintenance — Model Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('maintenance_results.png', dpi=150)
plt.show()
print("[OK] Results saved as 'maintenance_results.png'")

print("\n" + "=" * 60)
print("[OK] Lab 05 Complete!")
print("=" * 60)
