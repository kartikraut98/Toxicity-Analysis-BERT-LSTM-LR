# Multilabel Sentiment Classification

This repository implements a multilabel sentiment classification workflow using logistic regression with PyTorch. The model predicts multiple sentiment labels (e.g., toxicity, severe toxicity) for the Civil Comments dataset. The workflow includes data preprocessing, model training, evaluation, and model saving.

---

## Workflow Overview

### 1. **Data Loading and Preprocessing**
   - Uses the Civil Comments dataset (`google/civil_comments`) from Hugging Face.
   - Converts text data into TF-IDF features.
   - Prepares labels for multilabel classification by binarizing them (e.g., label = 1 if value > 0, else 0).
   - Creates PyTorch DataLoaders for batch processing.

### 2. **Model Architecture**
   - Implements a multilabel logistic regression model using PyTorch.
   - Features:
     - Sigmoid activation for multilabel probabilities.
     - Binary cross-entropy loss for optimization.
     - Support for training on CPU or GPU.

### 3. **Training**
   - Processes data in batches using a DataLoader.
   - Optimizes the model using the Adam optimizer.
   - Prints training progress for each epoch.

### 4. **Evaluation**
   - Generates predictions on the validation set.
   - Reports classification metrics such as precision, recall, and F1 score.

### 5. **Saving the Model**
   - Saves the trained model and TF-IDF vectorizer for reuse.

---

## Requirements

- Python 3.8+
- PyTorch 1.11+
- Hugging Face `datasets`
- `scikit-learn`
- `argparse`
- `pickle`
- CUDA-compatible GPU (optional)

Install dependencies:
```bash
pip install torch torchvision scikit-learn datasets tqdm

python sentiment_classifier.py --ngram 3 --max_features 10000 --epochs 100 --lr 0.01 --batch_size 64 --save_path ./sentiment_model.pkl --test_rows 1000
