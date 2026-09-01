import os
import sys
import glob
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_data(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Cleans numeric columns: handles infinite values and missing NaNs."""
    df = df.copy()
    for col in features:
        if col in df.columns:
            # Replace inf with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Impute NaN with median or 0
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
    return df

def normalize_cicids_attack_cat(label_str: str) -> str:
    """Normalizes raw CICIDS2017 string label into a clean category."""
    if not isinstance(label_str, str):
        return "Benign"
    label_lower = label_str.lower().strip()
    if "benign" in label_lower:
        return "Benign"
    elif "ddos" in label_lower:
        return "DDoS"
    elif "dos" in label_lower:
        return "DoS"
    elif "portscan" in label_lower:
        return "Portscan"
    elif "botnet" in label_lower or "bot" in label_lower:
        return "Botnet"
    elif "web" in label_lower:
        return "WebAttack"
    elif "bruteforce" in label_lower or "brute force" in label_lower or "patator" in label_lower:
        return "Bruteforce"
    elif "infiltration" in label_lower:
        return "Infiltration"
    else:
        return "Attack-Other"

def preprocess_cicids(raw_dir: str, features: list) -> pd.DataFrame:
    """Preprocesses CICIDS2017 Parquet files into a unified format."""
    logger.info(f"Starting CICIDS2017 preprocessing from: {raw_dir}")
    files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")
    
    dfs = []
    for f in files:
        logger.info(f"Reading file: {os.path.basename(f)}")
        df = pd.read_parquet(f, engine='pyarrow')
        
        # Mapping dict for CICIDS2017
        mapping = {
            'Flow Duration': 'duration',
            'Total Fwd Packets': 'src_packets',
            'Total Backward Packets': 'dst_packets',
            'Fwd Packets Length Total': 'src_bytes',
            'Bwd Packets Length Total': 'dst_bytes',
            'Flow Packets/s': 'rate',
            'Fwd Packet Length Mean': 'src_packet_mean',
            'Bwd Packet Length Mean': 'dst_packet_mean',
            'Init Fwd Win Bytes': 'src_win_bytes',
            'Init Bwd Win Bytes': 'dst_win_bytes'
        }
        
        # Extract features and target
        mapped_df = pd.DataFrame()
        for raw_col, unified_col in mapping.items():
            if raw_col in df.columns:
                mapped_df[unified_col] = df[raw_col].astype(float)
            else:
                logger.warning(f"Missing column {raw_col} in {f}, filling with 0")
                mapped_df[unified_col] = 0.0
                
        # Labels mapping
        if 'Label' in df.columns:
            mapped_df['attack_cat'] = df['Label'].apply(normalize_cicids_attack_cat)
            mapped_df['label'] = (mapped_df['attack_cat'] != 'Benign').astype(int)
        else:
            mapped_df['attack_cat'] = 'Benign'
            mapped_df['label'] = 0
            
        dfs.append(mapped_df)
        
    unified_df = pd.concat(dfs, ignore_index=True)
    unified_df = clean_data(unified_df, features)
    logger.info(f"Finished CICIDS2017 preprocessing. Shape: {unified_df.shape}")
    return unified_df

def preprocess_unsw(raw_dir: str, features: list) -> pd.DataFrame:
    """Preprocesses UNSW-NB15 Parquet files into a unified format."""
    logger.info(f"Starting UNSW-NB15 preprocessing from: {raw_dir}")
    files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")
        
    dfs = []
    for f in files:
        logger.info(f"Reading file: {os.path.basename(f)}")
        df = pd.read_parquet(f, engine='pyarrow')
        
        # Mapping dict for UNSW-NB15
        mapping = {
            'dur': 'duration',
            'spkts': 'src_packets',
            'dpkts': 'dst_packets',
            'sbytes': 'src_bytes',
            'dbytes': 'dst_bytes',
            'rate': 'rate',
            'smean': 'src_packet_mean',
            'dmean': 'dst_packet_mean',
            'swin': 'src_win_bytes',
            'dwin': 'dst_win_bytes'
        }
        
        mapped_df = pd.DataFrame()
        for raw_col, unified_col in mapping.items():
            if raw_col in df.columns:
                if raw_col == 'dur':
                    # Convert seconds to microseconds
                    mapped_df[unified_col] = df[raw_col].astype(float) * 1e6
                else:
                    mapped_df[unified_col] = df[raw_col].astype(float)
            else:
                logger.warning(f"Missing column {raw_col} in {f}, filling with 0")
                mapped_df[unified_col] = 0.0
                
        # Labels mapping
        if 'label' in df.columns:
            mapped_df['label'] = df['label'].astype(int)
        else:
            mapped_df['label'] = 0
        if 'attack_cat' in df.columns:
            # Clean string labels, map missing/nan to Benign
            cat_series = df['attack_cat'].astype(str).str.strip()
            cat_series = cat_series.replace({'nan': 'Benign', 'None': 'Benign', '': 'Benign'})
            mapped_df['attack_cat'] = cat_series
            mapped_df['attack_cat'] = mapped_df.apply(
                lambda row: 'Benign' if row['label'] == 0 else (row['attack_cat'] if row['attack_cat'] != 'Normal' else 'Attack-Other'), 
                axis=1
            )
        else:
            mapped_df['attack_cat'] = mapped_df['label'].apply(lambda x: 'Attack-Other' if x == 1 else 'Benign')
            
        dfs.append(mapped_df)
        
    unified_df = pd.concat(dfs, ignore_index=True)
    unified_df = clean_data(unified_df, features)
    logger.info(f"Finished UNSW-NB15 preprocessing. Shape: {unified_df.shape}")
    return unified_df

def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    features = config['data']['features']
    
    # 1. Preprocess CICIDS2017
    cic_df = preprocess_cicids(config['data']['raw_cic_dir'], features)
    cic_df.to_parquet(os.path.join(processed_dir, "cicids2017_unified.parquet"), index=False)
    
    # 2. Preprocess UNSW-NB15
    unsw_df = preprocess_unsw(config['data']['raw_unsw_dir'], features)
    unsw_df.to_parquet(os.path.join(processed_dir, "unsw_nb15_unified.parquet"), index=False)
    
    benign_cic_raw = cic_df[cic_df['label'] == 0][features].values
    if config['data'].get('log_transform', False):
        logger.info("Applying log1p transformation to scaler baseline data...")
        benign_cic_processed = np.log1p(np.maximum(0, benign_cic_raw))
    else:
        benign_cic_processed = benign_cic_raw
        
    scaler = StandardScaler()
    scaler.fit(benign_cic_processed)
    
    # Save the scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # 4. Train-Test Splits
    logger.info("Splitting CICIDS2017 dataset...")
    # Stratified split to maintain class balance in evaluation
    cic_train, cic_test = train_test_split(
        cic_df, 
        test_size=0.2, 
        stratify=cic_df['label'], 
        random_state=42
    )
    
    cic_train.to_parquet(os.path.join(processed_dir, "cicids2017_train.parquet"), index=False)
    cic_test.to_parquet(os.path.join(processed_dir, "cicids2017_test.parquet"), index=False)
    
    # Save UNSW-NB15 as a pure test/generalization dataset
    unsw_df.to_parquet(os.path.join(processed_dir, "unsw_nb15_test.parquet"), index=False)
    
    logger.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
