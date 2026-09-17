# Lab 10 — MLOps Workflow: Dockerfile
# Containerize the ML model for deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and scripts
COPY mlops_artifacts/ ./mlops_artifacts/
COPY train.py .

# Expose port for model serving
EXPOSE 5001

# Default command: run training pipeline
CMD ["python", "train.py"]
