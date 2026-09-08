"""
MLP Baseline Model for Network Intrusion Detection.
A simple feedforward neural network serving as the classical DL baseline.
Architecture: Input(10) → 128 → 64 → 32 → 1 (sigmoid)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class MLPNetwork(nn.Module):
    """Multi-Layer Perceptron for binary classification of network flows."""

    def __init__(self, input_dim: int = 10, hidden_dims: list = None, dropout: float = 0.3):
        super(MLPNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPClassifier:
    """Wrapper class for training, evaluating, and saving the MLP model."""

    def __init__(self, input_dim: int = 10, hidden_dims: list = None,
                 dropout: float = 0.3, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims else [128, 64, 32]
        self.dropout = dropout
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPNetwork(input_dim, self.hidden_dims, dropout).to(self.device)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
              batch_size: int = 256, val_split: float = 0.1):
        """Train the MLP model with optional validation monitoring."""
        # Validation split
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        logger.info(f"Training MLP on {len(train_idx)} samples for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t.to(self.device))
                val_loss = criterion(val_preds, y_val_t.to(self.device)).item()
            self.model.train()

            scheduler.step(val_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.is_trained = True
        logger.info("MLP training completed.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities."""
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = self.model(tensor_x)
            return probs.cpu().numpy().flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str):
        """Save the model state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'learning_rate': self.lr,
            'is_trained': self.is_trained
        }
        torch.save(state, filepath)
        logger.info(f"MLP model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MLPClassifier':
        """Load the model state."""
        state = torch.load(filepath, map_location=torch.device('cpu'))
        obj = cls(
            input_dim=state['input_dim'],
            hidden_dims=state['hidden_dims'],
            dropout=state['dropout'],
            learning_rate=state['learning_rate']
        )
        obj.model.load_state_dict(state['model_state_dict'])
        obj.is_trained = state['is_trained']
        logger.info(f"MLP model loaded from {filepath}")
        return obj
