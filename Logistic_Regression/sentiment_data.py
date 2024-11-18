from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import pickle


class SentimentData:
    def __init__(self, ngram_range=(1, 1), max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def load_data(self, limit_rows=None):
        dataset = load_dataset("google/civil_comments")

        label_columns = [
            'toxicity', 'severe_toxicity', 'obscene',
            'threat', 'insult', 'identity_attack', 'sexual_explicit'
        ]

        def binarize_labels(example):
            for label in label_columns:
                example[label] = 1 if example[label] > 0 else 0
            return example

        dataset = dataset.map(binarize_labels)

        if limit_rows:
            dataset = {split: dataset[split].select(range(limit_rows)) for split in dataset.keys()}

        return dataset['train'], dataset['validation'], dataset['test']

    def preprocess(self, dataset_split, label_columns, batch_size=16, device="cpu", limit_rows=None):
        df = dataset_split.to_pandas()
        if limit_rows:
            df = df.head(limit_rows)

        print("Converting text to TF-IDF features...")
        X = self.vectorizer.fit_transform(df['text'])

        # Save the vectorizer for reuse
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        # Convert TF-IDF matrix to sparse PyTorch tensors
        coo = X.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        sparse_X = torch.sparse_coo_tensor(indices, values, coo.shape)

        # Prepare labels
        Y = df[label_columns].values
        tensor_Y = torch.tensor(Y, dtype=torch.float32, device=device)

        # Custom collate function to handle sparse tensors
        def collate_fn(batch):
            features, labels = zip(*batch)
            dense_features = torch.stack([x.to_dense() for x in features])
            dense_labels = torch.stack(labels)
            return dense_features, dense_labels

        # Combine tensors into a DataLoader
        dataset = TensorDataset(sparse_X, tensor_Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
