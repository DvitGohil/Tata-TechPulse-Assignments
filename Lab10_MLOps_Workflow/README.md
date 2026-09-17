# Lab 10 — MLOps Workflow Simulation

## 🎯 Objective

Build a **CI/CD pipeline simulation** for ML model deployment using **MLflow** for experiment tracking and **Docker** for containerization. This lab demonstrates end-to-end MLOps practices.

---

## 📊 Dataset

- **Synthetic dataset** generated programmatically
- Simple regression task for demonstrating the MLOps pipeline
- Focus is on the workflow, not the model complexity

No manual download required.

---

## 🔧 Dependencies

```
numpy, pandas, scikit-learn, mlflow
```

For Docker containerization:
```
Docker Desktop (latest version)
```

---

## ▶️ How to Run

### Step 1: Run the Training Script with MLflow Tracking

```bash
cd Lab10_MLOps_Workflow
python train.py
```

This will:
- Train multiple models with different hyperparameters
- Log all experiments, metrics, and artifacts to MLflow
- Save the best model

### Step 2: View MLflow Dashboard (Optional)

```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser to view experiment results.

### Step 3: Build Docker Container (Optional)

```bash
docker build -t mlops-model .
docker run -p 5001:5001 mlops-model
```

---

## 📤 Expected Output

1. **Experiment Tracking** — Multiple model runs logged with hyperparameters
2. **Metrics Logging** — MAE, RMSE, R² for each experiment
3. **Model Registry** — Best model saved as MLflow artifact
4. **Comparison Table** — Side-by-side comparison of all runs
5. **Dockerfile** — Ready-to-build container for model serving

---

## 📝 Key Concepts

- MLflow Experiment Tracking
- Hyperparameter Logging
- Model Versioning & Registry
- Docker Containerization
- CI/CD Pipeline for ML
