import argparse
from model import MultilabelLogisticRegression
from sentiment_data import SentimentData
from sklearn.metrics import classification_report
from utils import save_model
import torch


def main():
    parser = argparse.ArgumentParser(description="Multilabel Sentiment Classification")
    parser.add_argument('--ngram', type=int, default=3, help="N-gram range (e.g., 1 for unigrams, 2 for bigrams, 3 for trigrams)")
    parser.add_argument('--max_features', type=int, default=10000, help="Maximum number of features for TF-IDF")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the model")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the trained model (optional)")
    parser.add_argument('--test_rows', type=int, default=1000, help="Number of rows to use for testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_columns = [
        'toxicity', 'severe_toxicity', 'obscene',
        'threat', 'insult', 'identity_attack', 'sexual_explicit'
    ]

    print("Loading and preprocessing data...")
    data_handler = SentimentData(ngram_range=(1, args.ngram), max_features=args.max_features)
    train_data, val_data, _ = data_handler.load_data(limit_rows=args.test_rows)

    train_loader = data_handler.preprocess(
        train_data, label_columns, batch_size=args.batch_size, device=device
    )
    val_loader = data_handler.preprocess(
        val_data, label_columns, batch_size=args.batch_size, device=device
    )

    print("Initializing the logistic regression model...")
    model = MultilabelLogisticRegression(lr=args.lr, epochs=args.epochs)

    print("Training the model...")
    model.fit(train_loader, device)

    print("Evaluating the model on validation set...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            preds = model.predict_classes(X_batch)
            all_preds.append(preds)
            all_labels.append(Y_batch)

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    print("Validation Performance Report:")
    print(classification_report(all_labels, all_preds, target_names=label_columns))

    if args.save_path:
        print(f"Saving the model to {args.save_path}...")
        save_model(model, args.save_path)


if __name__ == "__main__":
    main()
