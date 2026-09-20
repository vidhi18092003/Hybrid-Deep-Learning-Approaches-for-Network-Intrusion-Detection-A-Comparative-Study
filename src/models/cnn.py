"""
1D Convolutional Neural Network for Network Intrusion Detection.
Treats the 10 network flow features as a 1D signal and applies convolutional filters
to learn local feature interactions and spatial patterns.
Architecture: Conv1D(32) → Conv1D(64) → Conv1D(128) → GlobalAvgPool → FC → Sigmoid
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class CNN1DNetwork(nn.Module):
    """1D Convolutional Neural Network for binary classification of network flows.

    The 10 input features are treated as a 1D sequence of length 10 with 1 channel.
    Conv1D layers extract local feature interaction patterns.
    """

    def __init__(self, input_dim: int = 10, num_filters: list = None, dropout: float = 0.3):
        super(CNN1DNetwork, self).__init__()
        if num_filters is None:
            num_filters = [32, 64, 128]

        # Conv blocks: each is Conv1D → BatchNorm → ReLU → Dropout
        conv_layers = []
        in_channels = 1
        for n_filt in num_filters:
            conv_layers.append(nn.Conv1d(in_channels, n_filt, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm1d(n_filt))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = n_filt

        self.conv_block = nn.Sequential(*conv_layers)

        # Global Average Pooling followed by classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        # Reshape to [batch_size, 1, input_dim] for Conv1D (1 channel)
        x = x.unsqueeze(1)
        # Apply convolutions
        x = self.conv_block(x)  # [batch_size, num_filters[-1], input_dim]
        # Global Average Pooling across the spatial dimension
        x = x.mean(dim=2)  # [batch_size, num_filters[-1]]
        # Classify
        x = self.classifier(x)
        return x


class CNN1DClassifier:
    """Wrapper class for training, evaluating, and saving the 1D-CNN model."""

    def __init__(self, input_dim: int = 10, num_filters: list = None,
                 dropout: float = 0.3, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.num_filters = num_filters if num_filters else [32, 64, 128]
        self.dropout = dropout
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN1DNetwork(input_dim, self.num_filters, dropout).to(self.device)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
              batch_size: int = 256, val_split: float = 0.1):
        """Train the 1D-CNN model."""
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

        logger.info(f"Training 1D-CNN on {len(train_idx)} samples for {epochs} epochs...")
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

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t.to(self.device))
                val_loss = criterion(val_preds, y_val_t.to(self.device)).item()
            self.model.train()

            scheduler.step(val_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.is_trained = True
        logger.info("1D-CNN training completed.")

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
            'num_filters': self.num_filters,
            'dropout': self.dropout,
            'learning_rate': self.lr,
            'is_trained': self.is_trained
        }
        torch.save(state, filepath)
        logger.info(f"1D-CNN model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CNN1DClassifier':
        """Load the model state."""
        state = torch.load(filepath, map_location=torch.device('cpu'))
        obj = cls(
            input_dim=state['input_dim'],
            num_filters=state['num_filters'],
            dropout=state['dropout'],
            learning_rate=state['learning_rate']
        )
        obj.model.load_state_dict(state['model_state_dict'])
        obj.is_trained = state['is_trained']
        logger.info(f"1D-CNN model loaded from {filepath}")
        return obj
