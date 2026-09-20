import os
import pickle
import logging
import yaml
import pandas as pd
import numpy as np
import torch

from src.ingestion.preprocess import load_config
from src.models.kmeans import SpatialAnomalyFilter
from src.models.lstm import LSTMTemporalClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    log_transform_enabled = config['data'].get('log_transform', False)
    
    # 1. Load preprocessed datasets
    logger.info("Loading preprocessed training data...")
    train_path = os.path.join(processed_dir, "cicids2017_train.parquet")
    if not os.path.exists(train_path):
        logger.error(f"Training dataset not found at {train_path}. Please run preprocess.py first.")
        return
        
    df_train = pd.read_parquet(train_path)
    
    # Load Scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found at {scaler_path}. Please run preprocess.py first.")
        return
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Scale features
    X_train_raw = df_train[features].values
    if log_transform_enabled:
        logger.info("Applying log-transform to raw training features...")
        X_train_processed = np.log1p(np.maximum(0, X_train_raw))
    else:
        X_train_processed = X_train_raw
        
    X_train_scaled = scaler.transform(X_train_processed)
    y_train = df_train['label'].values
    
    # 2. Train Stage 1: Spatial Anomaly Filter (K-Means)
    # K-Means should only be trained on normal/benign traffic
    benign_mask = (y_train == 0)
    X_train_benign = X_train_scaled[benign_mask]
    
    kmeans_config = config['models']['kmeans']
    spatial_filter = SpatialAnomalyFilter(
        n_clusters=kmeans_config['n_clusters'],
        percentile_threshold=kmeans_config['percentile_threshold']
    )
    spatial_filter.fit(X_train_benign)
    spatial_filter.save(kmeans_config['save_path'])
    
    # 3. Train Stage 2: Temporal Classifier (GRU/LSTM)
    # The sequential classifier is trained on sequences of candidate anomalies flagged by K-Means
    logger.info("Flagging candidate anomalies for sequential model training...")
    candidate_mask = spatial_filter.predict_anomalies(X_train_scaled)
    
    X_train_candidates = X_train_scaled[candidate_mask]
    y_train_candidates = y_train[candidate_mask]
    
    benign_cand_idx = np.where(y_train_candidates == 0)[0]
    attack_cand_idx = np.where(y_train_candidates == 1)[0]
    
    logger.info(f"Filtered Candidates: Benign={len(benign_cand_idx)}, Attack={len(attack_cand_idx)}")
    
    # Balance the candidate dataset (Undersample the majority class)
    if len(benign_cand_idx) > 0 and len(attack_cand_idx) > 0:
        logger.info("Balancing candidate flows to a 1:1 ratio...")
        if len(attack_cand_idx) > len(benign_cand_idx):
            # Undersample attacks
            np.random.seed(42)
            np.random.shuffle(attack_cand_idx)
            keep_attack_idx = attack_cand_idx[:len(benign_cand_idx)]
            balanced_idx = np.concatenate([benign_cand_idx, keep_attack_idx])
        else:
            # Undersample benign
            np.random.seed(42)
            np.random.shuffle(benign_cand_idx)
            keep_benign_idx = benign_cand_idx[:len(attack_cand_idx)]
            balanced_idx = np.concatenate([keep_benign_idx, attack_cand_idx])
            
        np.random.shuffle(balanced_idx)
        X_train_balanced = X_train_candidates[balanced_idx]
        y_train_balanced = y_train_candidates[balanced_idx]
    else:
        logger.warning("No candidate flows found for one of the classes. Skipping balancing.")
        X_train_balanced = X_train_candidates
        y_train_balanced = y_train_candidates
        
    logger.info(f"Balanced Candidate Pool for Training: Benign={np.sum(y_train_balanced==0)}, Attack={np.sum(y_train_balanced==1)}")
    
    lstm_config = config['models']['lstm']
    temporal_classifier = LSTMTemporalClassifier(
        input_dim=lstm_config['input_dim'],
        hidden_dim=lstm_config['hidden_dim'],
        num_layers=lstm_config['num_layers'],
        dropout=lstm_config['dropout'],
        learning_rate=lstm_config['learning_rate'],
        seq_length=lstm_config['seq_length'],
        model_type=lstm_config.get('model_type', 'gru')
    )
    
    temporal_classifier.train(
        X=X_train_balanced,
        y=y_train_balanced,
        epochs=lstm_config['epochs'],
        batch_size=lstm_config['batch_size']
    )
    temporal_classifier.save(lstm_config['save_path'])
    logger.info("Base models training completed successfully.")

if __name__ == "__main__":
    main()
