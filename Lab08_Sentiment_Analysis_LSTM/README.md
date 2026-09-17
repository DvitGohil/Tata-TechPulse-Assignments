# Lab 08 — Sentiment Analysis using LSTM

## 🎯 Objective

Build an **LSTM (Long Short-Term Memory)** neural network to perform **sentiment analysis** on vehicle feedback/reviews. This lab demonstrates NLP techniques for text preprocessing and deep learning–based text classification.

---

## 📊 Dataset

- **Synthetic vehicle review dataset** generated programmatically
- Simulates customer feedback for automotive products/services
- Binary classification: Positive vs Negative sentiment
- ~1000 reviews with labeled sentiments

No manual download required.

---

## 🔧 Dependencies

```
numpy, pandas, matplotlib, tensorflow, keras, scikit-learn
```

---

## ▶️ How to Run

```bash
cd Lab08_Sentiment_Analysis_LSTM
python sentiment_analysis_lstm.py
```

---

## 📤 Expected Output

1. **Dataset Overview** — Sample reviews with labels
2. **Text Preprocessing** — Tokenization, padding, vocabulary stats
3. **LSTM Architecture** — Model summary
4. **Training Progress** — Accuracy/loss curves
5. **Evaluation** — Test accuracy, classification report
6. **Sample Predictions** — Predicted vs actual sentiment

---

## 📝 Key Concepts

- Text Preprocessing (Tokenization, Padding)
- Word Embeddings
- LSTM (Long Short-Term Memory) Networks
- Binary Sentiment Classification
- Overfitting Prevention (Dropout, Early Stopping)
