import torch
import numpy as np


class MultilabelLogisticRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.optimizer = None

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def initialize_weights(self, n_features, n_labels, device):
        # Initialize weights and bias for the model
        self.weights = torch.zeros((n_features, n_labels), device=device, requires_grad=True)
        self.bias = torch.zeros(n_labels, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.lr)

    def fit(self, data_loader, device):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for X_batch, Y_batch in data_loader:
                # Move data to the device (CPU/GPU)
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                # Initialize weights and bias on the first batch
                if self.weights is None:
                    n_features = X_batch.shape[1]
                    n_labels = Y_batch.shape[1]
                    self.initialize_weights(n_features, n_labels, device)

                # Forward pass
                logits = X_batch @ self.weights + self.bias
                preds = self.sigmoid(logits)
                loss = torch.nn.BCELoss()(preds, Y_batch)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = X.to(self.weights.device)
        logits = X @ self.weights + self.bias
        return self.sigmoid(logits)

    def predict_classes(self, X, threshold=0.01):
        probabilities = self.predict(X)
        return (probabilities > threshold).int()
