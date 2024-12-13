import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from utils import MultiLabelDataset, calculate_metrics, get_device
from config import label_columns, data_paths

# Initialize logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to log confusion matrix
def log_confusion_matrix(matrix, class_names, writer, tag, epoch):
    """
    Log a confusion matrix as an image in TensorBoard.
    
    Args:
        matrix (ndarray): The confusion matrix.
        class_names (list): List of class names.
        writer (SummaryWriter): TensorBoard writer object.
        tag (str): Tag for the image in TensorBoard.
        epoch (int): Epoch number for logging.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"{tag} Confusion Matrix")

    writer.add_figure(tag, fig, global_step=epoch)
    plt.close(fig)

# Function to log ROC curves
def log_roc_curves(preds, labels, label_names, writer, epoch):
    """
    Log ROC curves for each label to TensorBoard.

    Args:
        preds (ndarray): Predicted probabilities (N_samples, N_labels).
        labels (ndarray): Ground truth binary labels (N_samples, N_labels).
        label_names (list): Names of the labels.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Epoch for logging.
    """
    for i, label_name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC)")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        ax.set_title(f"ROC Curve - {label_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        # Log the ROC curve as an image
        writer.add_figure(f"ROC_Curve/{label_name}", fig, global_step=epoch)
        plt.close(fig)

def train_model():
    # Initialize TensorBoard writer
    writer = SummaryWriter("runs/roberta_multilabel_training")

    # Load datasets
    logging.info("Loading datasets...")
    train_df = pd.read_csv(data_paths['train'])
    val_df = pd.read_csv(data_paths['validation'])

    #train_df = pd.read_csv(data_paths['train']).iloc[:100]  # First 100 rows
    #val_df = pd.read_csv(data_paths['validation']).iloc[:50]  # First 50 rows


    # Load tokenizer and model
    logging.info("Initializing model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label_columns),
        problem_type="multi_label_classification"
    )
    device = get_device()
    model.to(device)

    # Create Datasets and DataLoaders
    logging.info("Creating datasets and dataloaders...")
    train_dataset = MultiLabelDataset(train_df, tokenizer, label_columns)
    val_dataset = MultiLabelDataset(val_df, tokenizer, label_columns)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}...")
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc="Training Batches"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            logits = outputs.logits
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        logging.info(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation Batches"):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )

                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels)

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        metrics = calculate_metrics(all_preds, all_labels, label_columns)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Metrics/Overall_ROC_AUC", metrics['overall_roc_auc'], epoch)
        writer.add_scalar("Metrics/Overall_Accuracy", metrics['overall_accuracy'], epoch)
        logging.info(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Metrics: {metrics}")

        # Log ROC curves and confusion matrix
        log_roc_curves(all_preds, all_labels, label_columns, writer, epoch)
        for i, label in enumerate(label_columns):
            cm = confusion_matrix(all_labels[:, i], (all_preds[:, i] > 0.5).astype(int))
            log_confusion_matrix(cm, ["Negative", "Positive"], writer, f"Confusion_Matrix/{label}", epoch)

    # Save the model
    logging.info("Saving the model...")
    model.save_pretrained("roberta_multilabel_model")
    tokenizer.save_pretrained("roberta_multilabel_model")
    writer.close()
    logging.info("Model saved to 'roberta_multilabel_model'")

if __name__ == "__main__":
    train_model()
