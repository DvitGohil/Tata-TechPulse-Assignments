"""
Lab 06 — Traffic Sign Classification using CNN
================================================
Train a CNN to recognize traffic signs using CIFAR-10 as a stand-in
for GTSRB dataset (to avoid large download requirements).

The architecture and techniques are identical to what would be used
with GTSRB — only the dataset differs for convenience.

Key Concepts:
- CNN Architecture (Conv2D, MaxPooling, Dropout)
- Image Preprocessing & Normalization
- Training Curves (Accuracy, Loss)
- Confusion Matrix for Multi-class Classification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 60)
print("LAB 06 — Traffic Sign Classification using CNN")
print("=" * 60)

# Using CIFAR-10 as a convenient stand-in for GTSRB
# CIFAR-10 classes serve as proxy "sign" categories
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\n[DATA] Training set: {X_train.shape}")
print(f"[DATA] Testing set:  {X_test.shape}")
print(f"[DATA] Image shape:  {X_train.shape[1:]}")
print(f"[DATA] Classes:      {len(class_names)}")

# ============================================================
# 2. Data Preprocessing
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Data Preprocessing")
print("-" * 40)

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("[OK] Images normalized to [0, 1]")
print("[OK] Labels one-hot encoded")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    ax = axes[i // 5][i % 5]
    idx = np.where(y_train.flatten() == i)[0][0]
    ax.imshow(X_train[idx])
    ax.set_title(class_names[i], fontweight='bold')
    ax.axis('off')
plt.suptitle('Sample Images — One Per Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150)
plt.show()
print("[OK] Sample images saved")

# ============================================================
# 3. Build CNN Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Build CNN Architecture")
print("-" * 40)

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 4. Train the Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Training the CNN")
print("-" * 40)

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.15,
    epochs=15,
    batch_size=64,
    verbose=1
)

# ============================================================
# 5. Evaluate the Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Model Evaluation")
print("-" * 40)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n[CHART] Test Accuracy: {test_acc:.4f}")
print(f"[CHART] Test Loss:     {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = y_test.flatten()

print(f"\n[INFO] Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ============================================================
# 6. Visualizations
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Training Accuracy
axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Training Loss
axes[1].plot(history.history['loss'], label='Train', linewidth=2, color='coral')
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='red')
axes[1].set_title('Model Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Confusion Matrix (subset for readability)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=class_names, yticklabels=class_names)
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_title('Confusion Matrix', fontweight='bold')
plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Traffic Sign CNN — Training Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_training_results.png', dpi=150)
plt.show()
print("[OK] Training results saved as 'cnn_training_results.png'")

# Sample predictions
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
sample_indices = np.random.choice(len(X_test), 10, replace=False)
for i, idx in enumerate(sample_indices):
    ax = axes[i // 5][i % 5]
    ax.imshow(X_test[idx])
    pred_label = class_names[y_pred[idx]]
    true_label = class_names[y_true[idx]]
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"P: {pred_label}\nT: {true_label}", color=color, fontsize=10)
    ax.axis('off')
plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150)
plt.show()
print("[OK] Sample predictions saved")

print("\n" + "=" * 60)
print("[OK] Lab 06 Complete!")
print("=" * 60)
