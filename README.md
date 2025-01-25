# Sentiment Analysis with Fine-Tuned BERT

This project demonstrates sentiment analysis using a fine-tuned **BERT model** (from Hugging Face's Transformers library) on the `moviereviews2.tsv` dataset. The goal was to compare the performance of a fine-tuned Transformer model with earlier experiments using VADER and Keras-based neural networks.

---

## Project Overview

### Objectives:
- Train a sentiment analysis model using **BERT**.
- Fine-tune the pre-trained `bert-base-uncased` model for binary classification (positive/negative sentiment).
- Compare the performance of BERT with earlier models.

### Dataset:
- Dataset: `moviereviews2.tsv`
- Format: Tab-separated values with two columns:
  - **review**: Text of the movie review.
  - **label**: Sentiment (1 for positive, 0 for negative).

---

## Methodology

### Steps:
1. **Data Preprocessing:**
   - Loaded the dataset using `pandas`.
   - Mapped sentiment labels (`pos`, `neg`) to integers (`1`, `0`).
   - Split the dataset into training (80%) and testing (20%) sets.

2. **Tokenization:**
   - Used `BertTokenizer` to tokenize the review texts.
   - Applied truncation and padding to ensure sequences were uniform in length (128 tokens).

3. **Model Definition:**
   - Used the pre-trained `bert-base-uncased` model.
   - Modified the classification head for binary sentiment classification (`num_labels=2`).

4. **Fine-Tuning:**
   - Fine-tuned the model using the following parameters:
     - Optimizer: `AdamW` with a learning rate of `2e-5`.
     - Batch size: `16`.
     - Epochs: `3`.
   - Used `cross-entropy loss` for optimization.

5. **Evaluation:**
   - Measured performance on the test set.
   - Compared accuracy and qualitative results with prior experiments.

---

## Key Results

- The fine-tuned BERT model achieved significantly better accuracy compared to:
  - **VADER** (which struggled with nuanced sentiment).
  - **Keras-based models** (trained on TF-IDF vectorized input).

---

## How to Run

### Install Dependencies:
```bash
pip install transformers torch pandas scikit-learn
```

### Run the Script:
1. Place the `moviereviews2.tsv` dataset in the project directory.
2. Execute the Python script (`sentiment_analysis_bert.py`).

---

## Example Code

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Example review
review = "I absolutely loved this movie!"

# Tokenize and prepare input
inputs = tokenizer(review, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

# Predict sentiment
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()

# Output result
print("Positive Sentiment" if predicted_class == 1 else "Negative Sentiment")
```

---

## Future Work

- Experiment with other Transformer models like **RoBERTa** and **DistilBERT**.
- Implement multi-class sentiment analysis.
- Fine-tune on larger, more diverse datasets for improved generalizability.

---
