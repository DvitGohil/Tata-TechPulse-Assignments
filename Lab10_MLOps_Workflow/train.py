"""
Lab 10 — MLOps Workflow Simulation
====================================
Build a CI/CD pipeline using MLflow for experiment tracking
and Docker for model deployment.

Key Concepts:
- MLflow Experiment Tracking
- Hyperparameter Logging
- Model Versioning
- Docker Containerization
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Try importing MLflow (optional dependency)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN]  MLflow not installed. Running without experiment tracking.")
    print("   Install with: pip install mlflow")

# ============================================================
# 1. Generate Synthetic Dataset
# ============================================================
print("=" * 60)
print("LAB 10 — MLOps Workflow Simulation")
print("=" * 60)

np.random.seed(42)
n = 1000

X = np.random.randn(n, 5)
feature_names = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E']
# True relationship: y = 3*A + 2*B - 1.5*C + 0.5*D + noise
y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.5, n)

df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"[LOOK] Features: {feature_names}")
print(df.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# ============================================================
# 2. Define Model Configurations
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Model Configurations")
print("-" * 40)

models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {'fit_intercept': True}
    },
    'Ridge_alpha_0.1': {
        'model': Ridge(alpha=0.1),
        'params': {'alpha': 0.1, 'solver': 'auto'}
    },
    'Ridge_alpha_1.0': {
        'model': Ridge(alpha=1.0),
        'params': {'alpha': 1.0, 'solver': 'auto'}
    },
    'Lasso_alpha_0.01': {
        'model': Lasso(alpha=0.01),
        'params': {'alpha': 0.01}
    },
    'RandomForest_100': {
        'model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'params': {'n_estimators': 100, 'max_depth': 10}
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                            learning_rate=0.1, random_state=42),
        'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
    },
}

print(f"[INFO] Models to train: {len(models)}")
for name in models:
    print(f"   - {name}")

# ============================================================
# 3. Train & Log Experiments
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Training & Experiment Tracking")
print("-" * 40)

# Create output directory for artifacts
os.makedirs('mlops_artifacts', exist_ok=True)

# Setup MLflow if available
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("Vehicle_Price_Prediction_Lab10")
    print("[OK] MLflow experiment set: 'Vehicle_Price_Prediction_Lab10'")

results = []

for name, config in models.items():
    print(f"\n[RUN] Training: {name}")

    model = config['model']
    params = config['params']

    # Train
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name):
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param('model_type', name)

            # Log metrics
            mlflow.log_metric('MAE', mae)
            mlflow.log_metric('RMSE', rmse)
            mlflow.log_metric('R2', r2)

            # Log model
            mlflow.sklearn.log_model(model, f"model_{name}")

            print(f"   [OK] Logged to MLflow (Run: {mlflow.active_run().info.run_id[:8]}...)")

    # Store results
    results.append({
        'Model': name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R²': round(r2, 4),
        'Params': params
    })

    print(f"   MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

# ============================================================
# 4. Results Comparison
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Results Comparison")
print("-" * 40)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)

print(f"\n[CHART] Model Comparison (sorted by R²):")
print(results_df[['Model', 'MAE', 'RMSE', 'R²']].to_string(index=False))

# Best model
best = results_df.iloc[0]
print(f"\n[BEST] Best Model: {best['Model']}")
print(f"   MAE:  {best['MAE']}")
print(f"   RMSE: {best['RMSE']}")
print(f"   R²:   {best['R²']}")

# ============================================================
# 5. Save Best Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Save Best Model")
print("-" * 40)

# Find and retrain best model
best_name = best['Model']
best_model = models[best_name]['model']
best_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(best_model, 'mlops_artifacts/best_model.pkl')
joblib.dump(scaler, 'mlops_artifacts/scaler.pkl')

# Save metadata
metadata = {
    'model_name': best_name,
    'metrics': {'MAE': best['MAE'], 'RMSE': best['RMSE'], 'R2': best['R²']},
    'features': feature_names,
    'params': best['Params'],
}
with open('mlops_artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

# Save results
results_df.to_csv('mlops_artifacts/experiment_results.csv', index=False)

print("[OK] Best model saved to 'mlops_artifacts/best_model.pkl'")
print("[OK] Scaler saved to 'mlops_artifacts/scaler.pkl'")
print("[OK] Metadata saved to 'mlops_artifacts/model_metadata.json'")
print("[OK] Results saved to 'mlops_artifacts/experiment_results.csv'")

# ============================================================
# 6. Pipeline Summary
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: MLOps Pipeline Summary")
print("-" * 40)

print("""
[INFO] MLOps Workflow Summary:
   ----------------------

   1. DATA PREPARATION
      +- Generated synthetic dataset (1000 samples, 5 features)

   2. EXPERIMENT TRACKING
      +- Trained 6 models with different configurations
      +- Logged parameters, metrics, and models to MLflow

   3. MODEL SELECTION
      +- Compared all models on MAE, RMSE, R²
      +- Selected best model automatically

   4. MODEL REGISTRY
      +- Saved best model, scaler, and metadata as artifacts
      +- Model versioning via MLflow (if available)

   5. CONTAINERIZATION
      +- Dockerfile provided for deployment
      +- Build: docker build -t mlops-model .
      +- Run:   docker run -p 5001:5001 mlops-model

   6. CI/CD PIPELINE (Conceptual)
      +- Code push Right Automated training Right Model evaluation
      +- If metrics pass threshold Right Deploy to production
      +- Monitor model performance in production
""")

# ============================================================
# 7. MLflow UI Instructions
# ============================================================
if MLFLOW_AVAILABLE:
    print("-" * 40)
    print("STEP 7: View Results in MLflow UI")
    print("-" * 40)
    print("""
    To view experiment results in the MLflow dashboard:

    1. Open terminal in this directory
    2. Run: mlflow ui
    3. Open: http://localhost:5000
    4. Compare models, metrics, and artifacts

    """)

print("=" * 60)
print("[OK] Lab 10 Complete!")
print("=" * 60)
