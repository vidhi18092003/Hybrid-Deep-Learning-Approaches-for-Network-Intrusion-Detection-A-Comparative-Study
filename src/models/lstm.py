import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class FlowLSTM(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(FlowLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :] # shape: [batch_size, hidden_dim]
        probs = self.sigmoid(self.fc(last_step_out))
        return probs

class FlowGRU(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(FlowGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        gru_out, _ = self.gru(x)
        last_step_out = gru_out[:, -1, :] # shape: [batch_size, hidden_dim]
        probs = self.sigmoid(self.fc(last_step_out))
        return probs

class FlowSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_length: int = 5):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        if len(self.features) >= self.seq_length:
            for i in range(len(self.features) - self.seq_length + 1):
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length
        seq_x = self.features[start_idx:end_idx]
        target_y = self.labels[end_idx - 1]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor([target_y], dtype=torch.float32)

class LSTMTemporalClassifier:
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, learning_rate: float = 0.001, seq_length: int = 5,
                 model_type: str = "gru"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.seq_length = seq_length
        self.model_type = model_type.lower()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_type == "gru":
            self.model = FlowGRU(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        else:
            self.model = FlowLSTM(input_dim, hidden_dim, num_layers, dropout).to(self.device)
            
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 128):
        """Trains the sequential model. Uses a standard BCELoss (data is balanced by the caller)."""
        if len(X) < self.seq_length:
            logger.warning("Not enough data to train sequential model. Skipping training.")
            return
            
        dataset = FlowSequenceDataset(X, y, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        logger.info(f"Training {self.model_type.upper()} on {len(dataset)} sequences for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                probs = self.model(batch_x)
                loss = criterion(probs, batch_y)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
        self.is_trained = True

    def fine_tune(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32):
        """Fine-tunes the model on a small batch of new streaming data (drift retraining)."""
        if not self.is_trained:
            self.train(X, y, epochs=epochs, batch_size=batch_size)
            return
            
        logger.info(f"Fine-tuning {self.model_type.upper()} on {len(X)} recent candidate flows...")
        original_lr = self.lr
        self.lr = original_lr * 0.1
        self.train(X, y, epochs=epochs, batch_size=batch_size)
        self.lr = original_lr

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Predicts anomaly probabilities for a given batch of sequences."""
        if not self.is_trained:
            logger.warning("Model is not trained yet. Returning random probabilities.")
            return np.random.uniform(0.01, 0.05, size=(len(X_seq), 1))
            
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
            probs = self.model(tensor_x)
            return probs.cpu().numpy()

    def predict_single_flow(self, recent_history: np.ndarray) -> float:
        """Classify the latest flow given the sequence of the last N candidate flows."""
        X_seq = np.expand_dims(recent_history, axis=0) # [1, seq_length, input_dim]
        prob = self.predict(X_seq)[0][0]
        return float(prob)

    def save(self, filepath: str):
        """Save the model parameters."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.lr,
            'seq_length': self.seq_length,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        torch.save(state, filepath)
        logger.info(f"Sequential model ({self.model_type.upper()}) saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LSTMTemporalClassifier':
        """Load the model parameters."""
        state = torch.load(filepath, map_location=torch.device('cpu'))
        obj = cls(
            input_dim=state['input_dim'],
            hidden_dim=state['hidden_dim'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
            learning_rate=state['learning_rate'],
            seq_length=state['seq_length'],
            model_type=state.get('model_type', 'lstm')
        )
        obj.model.load_state_dict(state['model_state_dict'])
        obj.is_trained = state['is_trained']
        logger.info(f"Sequential model ({obj.model_type.upper()}) loaded from {filepath}")
        return obj
