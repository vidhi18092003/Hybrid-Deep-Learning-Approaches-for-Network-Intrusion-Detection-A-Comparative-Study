import os
import time
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from src.ingestion.preprocess import load_config
from src.models.kmeans import SpatialAnomalyFilter
from src.models.lstm import LSTMTemporalClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineEvaluator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = self.config['data']['processed_dir']
        self.features = self.config['data']['features']
        self.seq_length = self.config['models']['lstm']['seq_length']
        self.log_transform_enabled = self.config['data'].get('log_transform', False)
        
        # Load Scaler
        scaler_path = os.path.join(self.processed_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load Stage 1 K-Means
        kmeans_path = self.config['models']['kmeans']['save_path']
        self.spatial_filter = SpatialAnomalyFilter.load(kmeans_path)
        
        # Load Stage 2 LSTM/GRU
        lstm_path = self.config['models']['lstm']['save_path']
        self.temporal_classifier = LSTMTemporalClassifier.load(lstm_path)

    def process_features(self, X_raw):
        """Applies log-transformation and scaling."""
        if self.log_transform_enabled:
            X_processed = np.log1p(np.maximum(0, X_raw))
        else:
            X_processed = X_raw
        return self.scaler.transform(X_processed)

    def evaluate_dataset(self, df: pd.DataFrame, name: str, custom_scaler=None, custom_kmeans=None, custom_classifier=None) -> dict:
        """Runs the hybrid anomaly detection pipeline over the dataset and computes metrics."""
        logger.info(f"\n--- Evaluating Hybrid Pipeline on {name} ---")
        total_flows = len(df)
        
        # 1. Processing features
        X_raw = df[self.features].values
        
        # Load custom models if provided (e.g. for adapted evaluation)
        scaler_obj = custom_scaler if custom_scaler else self.scaler
        kmeans_obj = custom_kmeans if custom_kmeans else self.spatial_filter
        classifier_obj = custom_classifier if custom_classifier else self.temporal_classifier
        
        if self.log_transform_enabled:
            X_processed = np.log1p(np.maximum(0, X_raw))
        else:
            X_processed = X_raw
        X_scaled = scaler_obj.transform(X_processed)
        y_true = df['label'].values
        
        # 2. Stage 1: Spatial Filter (K-Means)
        start_t = time.time()
        s1_candidates = kmeans_obj.predict_anomalies(X_scaled)
        s1_latency = (time.time() - start_t) * 1000 / total_flows
        
        s1_precision = precision_score(y_true, s1_candidates, zero_division=0)
        s1_recall = recall_score(y_true, s1_candidates, zero_division=0)
        s1_f1 = f1_score(y_true, s1_candidates, zero_division=0)
        
        logger.info(f"Stage 1 (K-Means Spatial Filter Only):")
        logger.info(f"  Candidates Flagged: {np.sum(s1_candidates)} / {total_flows} ({np.mean(s1_candidates)*100:.2f}%)")
        logger.info(f"  Precision: {s1_precision:.4f}, Recall: {s1_recall:.4f}, F1: {s1_f1:.4f}")
        logger.info(f"  Latency: {s1_latency:.4f} ms/flow")
        
        # 3. Stage 2: Sequential GRU/LSTM Classifier
        candidate_buffer = []
        hybrid_predictions = np.zeros(total_flows, dtype=int)
        hybrid_probabilities = np.zeros(total_flows)
        
        logger.info("Running sequential classification on candidates...")
        eval_start_t = time.time()
        
        for idx in range(total_flows):
            flow_scaled = X_scaled[idx]
            is_cand = s1_candidates[idx]
            
            if is_cand:
                # Add to candidate sequence buffer
                candidate_buffer.append(flow_scaled)
                if len(candidate_buffer) > self.seq_length:
                    candidate_buffer.pop(0)
                    
                if len(candidate_buffer) == self.seq_length:
                    seq_input = np.array(candidate_buffer)
                    prob = classifier_obj.predict_single_flow(seq_input)
                    hybrid_probabilities[idx] = prob
                    if prob > 0.5:
                        hybrid_predictions[idx] = 1
                else:
                    # In early steps when sequence buffer is not full, use default threshold prediction
                    hybrid_predictions[idx] = 1
                    hybrid_probabilities[idx] = 0.5
            else:
                # Filtered out by Stage 1 K-Means (classified as benign)
                hybrid_predictions[idx] = 0
                hybrid_probabilities[idx] = 0.0
                
        eval_total_t = time.time() - eval_start_t
        avg_hybrid_latency = (eval_total_t * 1000) / total_flows
        throughput = total_flows / eval_total_t
        
        # Compute hybrid metrics
        p_score = precision_score(y_true, hybrid_predictions, zero_division=0)
        r_score = recall_score(y_true, hybrid_predictions, zero_division=0)
        f1 = f1_score(y_true, hybrid_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, hybrid_probabilities)
        except ValueError:
            auc = 0.5
            
        logger.info(f"Hybrid Pipeline ({classifier_obj.model_type.upper()} Mode):")
        logger.info(f"  Precision: {p_score:.4f}, Recall: {r_score:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
        logger.info(f"  Average Latency: {avg_hybrid_latency:.4f} ms/flow")
        logger.info(f"  Throughput: {throughput:.2f} flows/sec")
        
        # Plot Confusion Matrix
        self.plot_confusion_matrix(y_true, hybrid_predictions, name)
        
        # Class Breakdown Performance
        if 'attack_cat' in df.columns:
            self.evaluate_by_attack_category(df, hybrid_predictions, name)
            
        return {
            'dataset': name,
            'precision': p_score, 'recall': r_score, 'f1': f1, 'auc': auc
        }

    def plot_confusion_matrix(self, y_true, y_pred, dataset_name: str):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        plt.title(f'Confusion Matrix - {dataset_name}\n(Hybrid Pipeline)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        filename = f"confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.processed_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {filepath}")

    def evaluate_by_attack_category(self, df: pd.DataFrame, y_pred: np.ndarray, dataset_name: str):
        """Prints and visualizes detection rate per attack category."""
        df_eval = df.copy()
        df_eval['pred'] = y_pred
        
        logger.info("\n--- Detection Performance by Attack Category ---")
        categories = df_eval['attack_cat'].unique()
        
        cat_performance = []
        for cat in categories:
            df_cat = df_eval[df_eval['attack_cat'] == cat]
            total = len(df_cat)
            detected = np.sum(df_cat['pred'] == df_cat['label']) if cat == 'Benign' else np.sum(df_cat['pred'] == 1)
            rate = detected / total if total > 0 else 0
            
            logger.info(f"  {cat:<18}: Total={total:<6} Detected={detected:<6} Rate={rate*100:.2f}%")
            cat_performance.append({
                'Category': cat,
                'Total': total,
                'Detected': detected,
                'Detection Rate (%)': rate * 100
            })
            
        df_perf = pd.DataFrame(cat_performance).sort_values(by='Detection Rate (%)', ascending=False)
        
        # Save to csv for dashboard display
        csv_path = os.path.join(self.processed_dir, f"category_performance_{dataset_name.lower().replace(' ', '_')}.csv")
        df_perf.to_csv(csv_path, index=False)
        
        # Plot Category Detection Rates
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_perf[df_perf['Category'] != 'Benign'], x='Detection Rate (%)', y='Category', hue='Category', palette='viridis', legend=False)
        plt.axvline(x=90, color='r', linestyle='--', alpha=0.7, label='90% Target')
        plt.title(f'Detection Rate per Attack Category - {dataset_name}')
        plt.xlabel('Detection Rate (%)')
        plt.ylabel('Attack Category')
        plt.xlim(0, 105)
        plt.tight_layout()
        
        filename = f"attack_cat_detection_{dataset_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.processed_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

    def run_adaptation_unsw(self, df_unsw: pd.DataFrame) -> tuple:
        """Simulates concept drift adaptation on UNSW-NB15 calibration slice."""
        logger.info("\n==================================================")
        logger.info("⚡ Simulating Concept Drift Adaptation on UNSW-NB15 ⚡")
        logger.info("==================================================")
        
        # Split into calibration and test
        df_cal = df_unsw.head(15000)
        df_test = df_unsw.tail(len(df_unsw) - 15000)
        
        X_cal_raw = df_cal[self.features].values
        
        # 1. Update Scaler on UNSW benign calibration flows
        benign_cal_mask = (df_cal['label'].values == 0)
        X_cal_benign_raw = X_cal_raw[benign_cal_mask]
        
        logger.info(f"Refitting StandardScaler on {len(X_cal_benign_raw)} UNSW benign flows...")
        if self.log_transform_enabled:
            X_cal_benign_processed = np.log1p(np.maximum(0, X_cal_benign_raw))
        else:
            X_cal_benign_processed = X_cal_benign_raw
            
        custom_scaler = StandardScaler()
        custom_scaler.fit(X_cal_benign_processed)
        
        # 2. Scale calibration data
        if self.log_transform_enabled:
            X_cal_processed = np.log1p(np.maximum(0, X_cal_raw))
        else:
            X_cal_processed = X_cal_raw
        X_cal_scaled = custom_scaler.transform(X_cal_processed)
        y_cal = df_cal['label'].values
        
        # 3. Partial fit K-Means
        logger.info("Adapting K-Means centroids to UNSW-NB15 benign traffic...")
        # Load a fresh K-Means model to adapt
        import copy
        custom_kmeans = copy.deepcopy(self.spatial_filter)
        custom_kmeans.partial_fit(X_cal_scaled[benign_cal_mask])
        
        # 4. Fine-tune Sequential GRU/LSTM
        logger.info("Flagging candidates on UNSW calibration set...")
        cal_cand_mask = custom_kmeans.predict_anomalies(X_cal_scaled)
        X_cal_cand = X_cal_scaled[cal_cand_mask]
        y_cal_cand = y_cal[cal_cand_mask]
        
        # Balance candidates
        benign_idx = np.where(y_cal_cand == 0)[0]
        attack_idx = np.where(y_cal_cand == 1)[0]
        
        if len(benign_idx) > 0 and len(attack_idx) > 0:
            if len(attack_idx) > len(benign_idx):
                np.random.shuffle(attack_idx)
                keep_idx = attack_idx[:len(benign_idx)]
                balanced_idx = np.concatenate([benign_idx, keep_idx])
            else:
                np.random.shuffle(benign_idx)
                keep_idx = benign_idx[:len(attack_idx)]
                balanced_idx = np.concatenate([keep_idx, attack_idx])
            np.random.shuffle(balanced_idx)
            X_cal_balanced = X_cal_cand[balanced_idx]
            y_cal_balanced = y_cal_cand[balanced_idx]
        else:
            X_cal_balanced = X_cal_cand
            y_cal_balanced = y_cal_cand
            
        custom_classifier = copy.deepcopy(self.temporal_classifier)
        if len(X_cal_balanced) >= self.seq_length:
            logger.info("Fine-tuning GRU weights on UNSW-NB15 candidate flows...")
            custom_classifier.fine_tune(X_cal_balanced, y_cal_balanced, epochs=2, batch_size=64)
            
        logger.info("UNSW-NB15 concept drift adaptation completed successfully.")
        return df_test, custom_scaler, custom_kmeans, custom_classifier

def main():
    evaluator = PipelineEvaluator()
    
    # 1. Evaluate on CICIDS2017 Test Set
    cic_test_path = os.path.join(evaluator.processed_dir, "cicids2017_test.parquet")
    if os.path.exists(cic_test_path):
        df_cic = pd.read_parquet(cic_test_path)
        if len(df_cic) > 50000:
            df_cic = df_cic.sample(50000, random_state=42).sort_index()
        evaluator.evaluate_dataset(df_cic, "CICIDS2017")
        
    # 2. Evaluate on UNSW-NB15 Test Set (Static/Drifted performance)
    unsw_test_path = os.path.join(evaluator.processed_dir, "unsw_nb15_test.parquet")
    if os.path.exists(unsw_test_path):
        df_unsw = pd.read_parquet(unsw_test_path)
        if len(df_unsw) > 50000:
            df_unsw = df_unsw.sample(50000, random_state=42).sort_index()
            
        # Static check
        evaluator.evaluate_dataset(df_unsw, "UNSW-NB15 (Static)")
        
        # Adaptation check (Drift-Adapted Performance)
        df_test_slice, custom_scaler, custom_kmeans, custom_classifier = evaluator.run_adaptation_unsw(df_unsw)
        evaluator.evaluate_dataset(df_test_slice, "UNSW-NB15 (Adapted)", custom_scaler, custom_kmeans, custom_classifier)

if __name__ == "__main__":
    main()
