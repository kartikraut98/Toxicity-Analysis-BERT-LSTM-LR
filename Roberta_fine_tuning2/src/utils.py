
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, hamming_loss, jaccard_score
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    """Get the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiLabelDataset(torch.utils.data.Dataset):
    """Custom Dataset for Multi-Label Classification."""
    def __init__(self, dataframe, tokenizer, label_columns, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        labels = torch.tensor(row[self.label_columns].astype(float).to_numpy(), dtype=torch.float32)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

def calculate_metrics(preds, labels, label_columns, threshold=0.5):
    """Calculate performance metrics for multi-label classification."""
    binarized_preds = (preds >= threshold).astype(int)
    binarized_labels = labels.astype(int)
    
    overall_accuracy = accuracy_score(binarized_labels.flatten(), binarized_preds.flatten())
    overall_roc_auc = roc_auc_score(binarized_labels, preds, average='macro')
    hamming = hamming_loss(binarized_labels, binarized_preds)
    jaccard = jaccard_score(binarized_labels, binarized_preds, average='macro')

    class_roc_auc = {
        label: roc_auc_score(binarized_labels[:, i], preds[:, i])
        for i, label in enumerate(label_columns)
    }

    return {
        'overall_accuracy': overall_accuracy,
        'overall_roc_auc': overall_roc_auc,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'class_roc_auc': class_roc_auc
    }
