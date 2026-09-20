"""
Unified Training Script for All 3 Deep Learning Models.
Trains MLP, 1D-CNN, and BiLSTM on the same preprocessed CICIDS2017 training data
with balanced classes and saves all models.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.preprocess import load_config
from src.models.mlp import MLPClassifier
from src.models.cnn import CNN1DClassifier
from src.models.bilstm import BiLSTMClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def balance_dataset(X: np.ndarray, y: np.ndarray, max_per_class: int = 50000) -> tuple:
    """Balance dataset by undersampling the majority class."""
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    logger.info(f"Original class distribution: Benign={len(benign_idx)}, Attack={len(attack_idx)}")

    min_count = min(len(benign_idx), len(attack_idx), max_per_class)

    np.random.seed(42)
    if len(benign_idx) > min_count:
        benign_idx = np.random.choice(benign_idx, min_count, replace=False)
    if len(attack_idx) > min_count:
        attack_idx = np.random.choice(attack_idx, min_count, replace=False)

    balanced_idx = np.concatenate([benign_idx, attack_idx])
    np.random.shuffle(balanced_idx)

    logger.info(f"Balanced class distribution: Benign={len(benign_idx)}, Attack={len(attack_idx)}")
    return X[balanced_idx], y[balanced_idx]


def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    log_transform = config['data'].get('log_transform', False)

    # ── 1. Load Training Data ──
    train_path = os.path.join(processed_dir, "cicids2017_train.parquet")
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}. Run preprocess.py first.")
        return

    logger.info("Loading preprocessed training data...")
    df_train = pd.read_parquet(train_path)

    # Load scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Scale features
    X_raw = df_train[features].values
    if log_transform:
        X_processed = np.log1p(np.maximum(0, X_raw))
    else:
        X_processed = X_raw

    X_scaled = scaler.transform(X_processed)
    y = df_train['label'].values

    # Balance dataset
    X_balanced, y_balanced = balance_dataset(X_scaled, y)

    # ── 2. Model Configurations ──
    mlp_config = config.get('models', {}).get('mlp', {})
    cnn_config = config.get('models', {}).get('cnn', {})
    bilstm_config = config.get('models', {}).get('bilstm', {})

    # ── 3. Train MLP ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL 1/3: MLP BASELINE")
    logger.info("=" * 60)

    mlp = MLPClassifier(
        input_dim=mlp_config.get('input_dim', 10),
        hidden_dims=mlp_config.get('hidden_dims', [128, 64, 32]),
        dropout=mlp_config.get('dropout', 0.3),
        learning_rate=mlp_config.get('learning_rate', 0.001)
    )
    mlp.train(
        X_balanced, y_balanced,
        epochs=mlp_config.get('epochs', 20),
        batch_size=mlp_config.get('batch_size', 256)
    )
    mlp_save_path = mlp_config.get('save_path', os.path.join(processed_dir, "mlp_model.pt"))
    mlp.save(mlp_save_path)

    # ── 4. Train 1D-CNN ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL 2/3: 1D-CNN")
    logger.info("=" * 60)

    cnn = CNN1DClassifier(
        input_dim=cnn_config.get('input_dim', 10),
        num_filters=cnn_config.get('num_filters', [32, 64, 128]),
        dropout=cnn_config.get('dropout', 0.3),
        learning_rate=cnn_config.get('learning_rate', 0.001)
    )
    cnn.train(
        X_balanced, y_balanced,
        epochs=cnn_config.get('epochs', 20),
        batch_size=cnn_config.get('batch_size', 256)
    )
    cnn_save_path = cnn_config.get('save_path', os.path.join(processed_dir, "cnn_model.pt"))
    cnn.save(cnn_save_path)

    # ── 5. Train BiLSTM ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL 3/3: BiLSTM")
    logger.info("=" * 60)

    bilstm = BiLSTMClassifier(
        input_dim=bilstm_config.get('input_dim', 10),
        hidden_dim=bilstm_config.get('hidden_dim', 64),
        num_layers=bilstm_config.get('num_layers', 2),
        dropout=bilstm_config.get('dropout', 0.3),
        learning_rate=bilstm_config.get('learning_rate', 0.001),
        seq_length=bilstm_config.get('seq_length', 5)
    )
    bilstm.train(
        X_balanced, y_balanced,
        epochs=bilstm_config.get('epochs', 15),
        batch_size=bilstm_config.get('batch_size', 128)
    )
    bilstm_save_path = bilstm_config.get('save_path', os.path.join(processed_dir, "bilstm_model.pt"))
    bilstm.save(bilstm_save_path)

    logger.info("\n" + "=" * 60)
    logger.info("ALL 3 MODELS TRAINED AND SAVED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
