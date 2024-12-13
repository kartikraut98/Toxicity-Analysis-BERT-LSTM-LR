
import pandas as pd
from sklearn.utils import resample
from datasets import load_dataset
from config import label_columns, thresholds

def binarize_labels(dataframe, thresholds):
    """Binarize labels in the dataset based on provided thresholds."""
    for label, threshold in thresholds.items():
        dataframe[label] = pd.to_numeric(dataframe[label], errors='coerce').fillna(0)
        dataframe[label] = (dataframe[label] > threshold).astype(int)
    return dataframe

def create_balanced_dataset(dataframe, label_columns, min_samples_per_class=5000, negative_sample_ratio=0.5):
    """Create a balanced dataset for multi-label classification."""
    dataframe[label_columns] = dataframe[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    balanced_df = []

    for label in label_columns:
        positive_samples = dataframe[dataframe[label] == 1]
        if len(positive_samples) < min_samples_per_class:
            positive_samples = resample(
                positive_samples,
                replace=True,
                n_samples=min_samples_per_class,
                random_state=42
            )
        else:
            positive_samples = positive_samples.sample(n=min_samples_per_class, random_state=42)
        balanced_df.append(positive_samples)

    balanced_df = pd.concat(balanced_df).drop_duplicates()

    negative_samples = dataframe[dataframe[label_columns].sum(axis=1) == 0]
    num_negative_samples = int(len(balanced_df) * negative_sample_ratio)
    if len(negative_samples) < num_negative_samples:
        negative_samples = resample(
            negative_samples,
            replace=True,
            n_samples=num_negative_samples,
            random_state=42
        )
    else:
        negative_samples = negative_samples.sample(n=num_negative_samples, random_state=42)

    balanced_df = pd.concat([balanced_df, negative_samples]).drop_duplicates()
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    dataset = load_dataset("google/civil_comments")

    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas()
    test_df = dataset['test'].to_pandas()

    train_df = binarize_labels(train_df, thresholds)
    val_df = binarize_labels(val_df, thresholds)
    test_df = binarize_labels(test_df, thresholds)

    train_balanced = create_balanced_dataset(train_df, label_columns)
    val_balanced = create_balanced_dataset(val_df, label_columns, min_samples_per_class=1000)
    test_balanced = create_balanced_dataset(test_df, label_columns, min_samples_per_class=1000)

    train_balanced.to_csv("../data/train_balanced.csv", index=False)
    val_balanced.to_csv("../data/val_balanced.csv", index=False)
    test_balanced.to_csv("../data/test_balanced.csv", index=False)
    print("Balanced datasets saved.")
