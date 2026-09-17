"""
Lab 08 — Sentiment Analysis using LSTM
========================================
Analyze vehicle feedback using LSTM-based sentiment classification.

Key Concepts:
- Text Preprocessing (Tokenization, Padding)
- Word Embeddings
- LSTM Networks
- Binary Sentiment Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================
# 1. Generate Synthetic Vehicle Review Dataset
# ============================================================
print("=" * 60)
print("LAB 08 — Sentiment Analysis using LSTM")
print("=" * 60)

np.random.seed(42)

# Positive review templates
positive_templates = [
    "The car has excellent mileage and smooth driving experience",
    "Great build quality and comfortable seats for long drives",
    "Amazing safety features and reliable engine performance",
    "Love the infotainment system and modern dashboard design",
    "Fuel efficiency is outstanding and maintenance cost is low",
    "Smooth gear shifting and powerful acceleration on highways",
    "Best in class suspension and comfortable ride quality",
    "Excellent after sales service and responsive customer support",
    "Premium interiors with spacious cabin and large boot space",
    "Value for money with loaded features and good resale value",
    "Powerful engine with great pickup and top speed performance",
    "Impressive looks and aerodynamic design turns heads everywhere",
    "Advanced technology features and connected car experience",
    "Quiet cabin with excellent noise insulation and refinement",
    "Safe and reliable car with five star crash test rating",
    "Easy to drive in city traffic with light steering",
    "Good ground clearance and handles rough roads perfectly",
    "Responsive brakes and stable handling at high speeds",
    "Well designed dashboard with user friendly controls layout",
    "Comfortable rear seat space perfect for family trips",
]

# Negative review templates
negative_templates = [
    "Poor mileage and very high maintenance cost for this car",
    "Engine makes too much noise and vibration at high speeds",
    "Bad after sales service and dealership is not responsive",
    "Uncomfortable seats cause back pain on long distance drives",
    "Frequent breakdowns and reliability is a major concern",
    "Suspension is too stiff and ride quality is very poor",
    "Outdated infotainment system with slow and laggy interface",
    "Paint quality is poor and starts rusting within two years",
    "Gear shifting is rough and clutch is too hard to press",
    "Not worth the price considering the features you get",
    "Air conditioning is weak and takes forever to cool cabin",
    "Brake performance is below average and feels unsafe",
    "Boot space is too small for a family car this size",
    "High fuel consumption makes it expensive to run daily",
    "Cheap plastic interiors look and feel very low quality",
    "Electrical problems keep occurring and wiring is unreliable",
    "Steering feels very heavy especially while parking",
    "No safety airbags in base variant which is disappointing",
    "Engine overheating issues in summer and hot climate conditions",
    "Resale value drops significantly making it a bad investment",
]

# Add variety with slight modifications
def generate_reviews(templates, n_reviews, label):
    reviews = []
    adjectives_pos = ['excellent', 'amazing', 'wonderful', 'fantastic', 'superb', 'great']
    adjectives_neg = ['terrible', 'awful', 'disappointing', 'horrible', 'worst', 'bad']

    for i in range(n_reviews):
        review = templates[i % len(templates)]
        # Add some variety
        if np.random.random() > 0.5:
            if label == 1:
                adj = np.random.choice(adjectives_pos)
                review = f"{adj} vehicle experience {review.lower()}"
            else:
                adj = np.random.choice(adjectives_neg)
                review = f"{adj} vehicle experience {review.lower()}"
        reviews.append(review)
    return reviews


positive_reviews = generate_reviews(positive_templates, 500, 1)
negative_reviews = generate_reviews(negative_templates, 500, 0)

all_reviews = positive_reviews + negative_reviews
all_labels = [1] * 500 + [0] * 500

df = pd.DataFrame({'review': all_reviews, 'sentiment': all_labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n[DATA] Dataset Shape: {df.shape}")
print(f"\n[CHART] Class Distribution:")
print(df['sentiment'].value_counts().rename({1: 'Positive', 0: 'Negative'}))
print(f"\n[LOOK] Sample Reviews:")
for i in range(3):
    label = '[OK] Positive' if df.iloc[i]['sentiment'] == 1 else '[X] Negative'
    print(f"  [{label}] {df.iloc[i]['review'][:80]}...")

# ============================================================
# 2. Text Preprocessing
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Text Preprocessing")
print("-" * 40)

# Tokenization
MAX_WORDS = 5000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])

# Pad sequences
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = np.array(df['sentiment'])

print(f"\n[NOTE] Vocabulary Size: {len(tokenizer.word_index)}")
print(f"[NOTE] Max Sequence Length: {MAX_LEN}")
print(f"[NOTE] Padded Shape: {X.shape}")

# Show top words
word_counts = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
print(f"\n[CHART] Top 10 Words: {[w[0] for w in word_counts[:10]]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# ============================================================
# 3. Build LSTM Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Build LSTM Architecture")
print("-" * 40)

EMBEDDING_DIM = 64

model = Sequential([
    Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 4. Train the Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Training the LSTM")
print("-" * 40)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=15,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# 5. Evaluate the Model
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Model Evaluation")
print("-" * 40)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[CHART] Test Accuracy: {test_acc:.4f}")
print(f"[CHART] Test Loss:     {test_loss:.4f}")

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print(f"\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ============================================================
# 6. Visualizations
# ============================================================
print("\n" + "-" * 40)
print("STEP 6: Visualizations")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

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

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_title('Confusion Matrix', fontweight='bold')

plt.suptitle('Sentiment Analysis LSTM — Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sentiment_results.png', dpi=150)
plt.show()
print("[OK] Results saved as 'sentiment_results.png'")

# ============================================================
# 7. Sample Predictions
# ============================================================
print("\n" + "-" * 40)
print("STEP 7: Sample Predictions")
print("-" * 40)

test_reviews = df.iloc[X_test.shape[0]:X_test.shape[0] + 5]['review'].tolist()
if len(test_reviews) < 5:
    test_reviews = [
        "This car is absolutely amazing with great features",
        "Terrible experience and very poor build quality",
        "Smooth ride and excellent fuel efficiency overall",
        "Worst service center experience with rude staff",
        "Perfect family car with spacious interiors and comfort",
    ]

for i, review in enumerate(test_reviews):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    sentiment = 'Positive [OK]' if prob > 0.5 else 'Negative [X]'
    print(f"\n  Review: \"{review[:70]}...\"")
    print(f"  Prediction: {sentiment} (confidence: {prob:.4f})")

print("\n" + "=" * 60)
print("[OK] Lab 08 Complete!")
print("=" * 60)
