
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from scripts.utils import MultiLabelDataset, calculate_metrics, get_device
from scripts.config import label_columns, data_paths

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model():
    # Load test dataset
    logging.info("Loading test dataset...")
    test_df = pd.read_csv(data_paths['test'])

    # Load tokenizer and model
    logging.info("Loading model and tokenizer...")
    model = RobertaForSequenceClassification.from_pretrained("roberta_multilabel_model")
    tokenizer = RobertaTokenizer.from_pretrained("roberta_multilabel_model")
    device = get_device()
    model.to(device)

    # Create Dataset and DataLoader
    logging.info("Creating test dataset and dataloader...")
    test_dataset = MultiLabelDataset(test_df, tokenizer, label_columns)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Evaluate the model
    logging.info("Evaluating the model...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Batches"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    # Combine predictions and labels
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels, label_columns)
    logging.info("Test Metrics:")
    logging.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    logging.info(f"Overall ROC-AUC: {metrics['overall_roc_auc']:.4f}")
    logging.info(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    logging.info(f"Jaccard Score: {metrics['jaccard_score']:.4f}")

    for label, roc_auc in metrics['class_roc_auc'].items():
        logging.info(f"ROC-AUC for {label}: {roc_auc:.4f}")

    # Save predictions
    logging.info("Saving predictions...")
    predictions_df = pd.DataFrame(all_preds, columns=label_columns)
    output_df = pd.concat([test_df['text'], predictions_df], axis=1)
    output_df.to_csv("test_predictions.csv", index=False)
    logging.info("Predictions saved to 'test_predictions.csv'")

if __name__ == "__main__":
    test_model()
