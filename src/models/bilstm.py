"""
Bidirectional LSTM Model for Network Intrusion Detection.
Uses a BiLSTM to capture both forward and backward temporal dependencies
in sequences of network flow features.
Architecture: BiLSTM(2 layers, hidden=64) → Concat(fwd+bwd) → FC(128→64→1) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class BiLSTMNetwork(nn.Module):
    """Bidirectional LSTM network for binary classification of network flow sequences."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super(BiLSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # BiLSTM outputs 2*hidden_dim (forward + backward)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.bilstm(x)
        # Take the last time step output (contains both forward and backward info)
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]
        return self.classifier(last_output)


class FlowSequenceDataset(Dataset):
    """Dataset that creates sliding-window sequences from flow data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_length: int = 5):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        self.indices = []
        if len(self.features) >= self.seq_length:
            for i in range(len(self.features) - self.seq_length + 1):
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seq_length
        seq_x = self.features[start:end]
        target_y = self.labels[end - 1]
        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor([target_y], dtype=torch.float32))


class BiLSTMClassifier:
    """Wrapper class for training, evaluating, and saving the BiLSTM model."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 learning_rate: float = 0.001, seq_length: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.seq_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMNetwork(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 15,
              batch_size: int = 128, val_split: float = 0.1):
        """Train the BiLSTM model on sequences of flows."""
        if len(X) < self.seq_length:
            logger.warning("Not enough data to train BiLSTM. Skipping.")
            return

        # Create sequence datasets — split before sequencing to avoid data leakage
        n_val = int(len(X) * val_split)
        X_train, y_train = X[n_val:], y[n_val:]
        X_val, y_val = X[:n_val], y[:n_val]

        train_ds = FlowSequenceDataset(X_train, y_train, self.seq_length)
        val_ds = FlowSequenceDataset(X_val, y_val, self.seq_length)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        logger.info(f"Training BiLSTM on {len(train_ds)} sequences for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / max(len(train_loader), 1)

            # Validation
            self.model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(self.device), vy.to(self.device)
                    vp = self.model(vx)
                    val_loss_total += criterion(vp, vy).item()
                    val_batches += 1
            val_loss = val_loss_total / max(val_batches, 1)
            self.model.train()

            scheduler.step(val_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.is_trained = True
        logger.info("BiLSTM training completed.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for each flow in the dataset.

        Creates sequences using a sliding window and assigns the prediction
        to the last element of each window.
        """
        if not self.is_trained:
            logger.warning("BiLSTM not trained. Returning zeros.")
            return np.zeros(len(X))

        self.model.eval()
        predictions = np.zeros(len(X))

        if len(X) < self.seq_length:
            return predictions

        # Vectorized sliding window creation for 1000x faster inference
        seqs = np.lib.stride_tricks.sliding_window_view(X, (self.seq_length, X.shape[1])).squeeze(1)

        all_probs = []
        batch_size = 2048
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i + batch_size]
                tensor_x = torch.tensor(batch_seqs, dtype=torch.float32).to(self.device)
                probs = self.model(tensor_x).cpu().numpy().flatten()
                all_probs.append(probs)

        probs_arr = np.concatenate(all_probs)
        predictions[self.seq_length - 1:] = probs_arr
        predictions[:self.seq_length - 1] = probs_arr[0]

        return predictions

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
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.lr,
            'seq_length': self.seq_length,
            'is_trained': self.is_trained
        }
        torch.save(state, filepath)
        logger.info(f"BiLSTM model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BiLSTMClassifier':
        """Load the model state."""
        state = torch.load(filepath, map_location=torch.device('cpu'))
        obj = cls(
            input_dim=state['input_dim'],
            hidden_dim=state['hidden_dim'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
            learning_rate=state['learning_rate'],
            seq_length=state['seq_length']
        )
        obj.model.load_state_dict(state['model_state_dict'])
        obj.is_trained = state['is_trained']
        logger.info(f"BiLSTM model loaded from {filepath}")
        return obj
