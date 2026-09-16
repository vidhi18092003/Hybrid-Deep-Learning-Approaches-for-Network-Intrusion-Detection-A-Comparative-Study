import os
import sys
import time
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.preprocess import load_config
from src.models.kmeans import SpatialAnomalyFilter
from src.models.lstm import LSTMTemporalClassifier
from src.drift.detector import DriftDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedStreamingConsumer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = self.config['data']['processed_dir']
        self.features = self.config['data']['features']
        self.seq_length = self.config['models']['lstm']['seq_length']
        self.local_stream_dir = self.config['streaming']['local_stream_dir']
        self.log_transform_enabled = self.config['data'].get('log_transform', False)
        self.output_file = os.path.join(self.processed_dir, "live_results.json")
        
        # Ensure target directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.local_stream_dir, exist_ok=True)
        
        # Load scaler
        scaler_path = os.path.join(self.processed_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load Stage 1 Spatial Filter
        kmeans_path = self.config['models']['kmeans']['save_path']
        self.spatial_filter = SpatialAnomalyFilter.load(kmeans_path)
        
        # Load Stage 2 GRU/LSTM Classifier
        lstm_path = self.config['models']['lstm']['save_path']
        self.temporal_classifier = LSTMTemporalClassifier.load(lstm_path)
        
        # Load historical benign data to initialize the drift detector reference
        train_path = os.path.join(self.processed_dir, "cicids2017_train.parquet")
        df_train = pd.read_parquet(train_path)
        X_train_benign_raw = df_train[df_train['label'] == 0][self.features].values
        
        # Log-transform reference data
        if self.log_transform_enabled:
            X_train_benign_processed = np.log1p(np.maximum(0, X_train_benign_raw))
        else:
            X_train_benign_processed = X_train_benign_raw
            
        X_train_benign_scaled = self.scaler.transform(X_train_benign_processed)
        
        # Subsample for reference size (e.g. 5000)
        ref_size = min(len(X_train_benign_scaled), self.config['drift']['reference_size'])
        indices = np.random.choice(len(X_train_benign_scaled), ref_size, replace=False)
        self.reference_data = X_train_benign_scaled[indices]
        
        # Initialize Concept Drift Detector
        self.drift_detector = DriftDetector(
            reference_data=self.reference_data,
            alpha=self.config['drift']['alpha'],
            min_drift_features=self.config['drift']['min_drift_features'],
            window_size=self.config['drift']['window_size']
        )
        
        # Buffers for online prediction
        self.candidate_buffer = [] # Store scaled features of candidates for sequence input
        self.all_recent_flows = []  # Buffer for retraining window
        self.max_retrain_size = self.config['drift']['retrain_window_size']
        
        # Metrics for dashboard
        self.processed_count = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        self.latency_sum = 0.0
        
        # Setup output results file
        if os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except Exception:
                pass
        with open(self.output_file, 'w') as f:
            f.write("")

    def process_batch(self, batch_data: list):
        """Processes a batch of flows arriving in the stream."""
        if not batch_data:
            return
            
        df_batch = pd.DataFrame(batch_data)
        
        # Extract features, log-transform and scale
        X_raw = df_batch[self.features].values
        if self.log_transform_enabled:
            X_processed = np.log1p(np.maximum(0, X_raw))
        else:
            X_processed = X_raw
        X_scaled = self.scaler.transform(X_processed)
        
        results_to_write = []
        
        for idx, row in df_batch.iterrows():
            flow_start_t = time.time()
            flow_scaled = X_scaled[idx]
            
            # Step 1: Spatial Filter (K-Means)
            flow_scaled_2d = np.expand_dims(flow_scaled, axis=0)
            is_candidate = self.spatial_filter.predict_anomalies(flow_scaled_2d)[0]
            
            anomaly_prob = 0.0
            predicted_label = 0
            filter_status = "Benign"
            
            if is_candidate:
                filter_status = "Flagged Candidate"
                self.candidate_buffer.append(flow_scaled)
                
                # Maintain sequence window
                if len(self.candidate_buffer) > self.seq_length:
                    self.candidate_buffer.pop(0)
                    
                # Step 2: Temporal Classifier
                if len(self.candidate_buffer) == self.seq_length:
                    seq_input = np.array(self.candidate_buffer) # Shape [seq_len, input_dim]
                    anomaly_prob = self.temporal_classifier.predict_single_flow(seq_input)
                    if anomaly_prob > 0.5:
                        predicted_label = 1
                        filter_status = "Confirmed Anomaly"
                        self.anomaly_count += 1
                    else:
                        filter_status = "Filtered Benign"
                else:
                    # Buffer not full, conservative flag
                    anomaly_prob = 0.5
                    predicted_label = 1
            
            # Calculate Latency
            latency_ms = (time.time() - flow_start_t) * 1000
            self.latency_sum += latency_ms
            self.processed_count += 1
            
            # Save flow to sliding window for drift detection and retraining
            self.all_recent_flows.append({
                'features': flow_scaled,
                'label': int(row.get('label', 0)),
                'is_benign': int(row.get('label', 0)) == 0
            })
            if len(self.all_recent_flows) > self.max_retrain_size:
                self.all_recent_flows.pop(0)
                
            # Drift Check
            drift_detected = self.drift_detector.add_samples([flow_scaled])
            
            # Prepare result entry
            result_entry = {
                'timestamp': int(time.time() * 1000),
                'duration': float(row['duration']),
                'src_packets': float(row['src_packets']),
                'dst_packets': float(row['dst_packets']),
                'src_bytes': float(row['src_bytes']),
                'dst_bytes': float(row['dst_bytes']),
                'rate': float(row['rate']),
                'attack_cat': str(row.get('attack_cat', 'Benign')),
                'true_label': int(row.get('label', 0)),
                'spatial_distance': float(self.spatial_filter.compute_distances(flow_scaled_2d)[0]),
                'spatial_threshold': float(self.spatial_filter.threshold),
                'anomaly_prob': float(anomaly_prob),
                'prediction': int(predicted_label),
                'status': filter_status,
                'latency_ms': float(latency_ms),
                'drift_triggered': bool(drift_detected),
                'processed_total': int(self.processed_count),
                'anomalies_total': int(self.anomaly_count),
                'throughput': float(self.processed_count / max(0.001, time.time() - self.start_time))
            }
            results_to_write.append(result_entry)
            
            if drift_detected:
                logger.warning("Triggering sliding-window retraining pipeline due to concept drift...")
                self.trigger_retraining()
                
        # Append batch results
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for item in results_to_write:
                f.write(json.dumps(item) + "\n")
            f.flush()

    def trigger_retraining(self):
        """Retrains both K-Means and GRU/LSTM models on recent balanced streaming window."""
        if len(self.all_recent_flows) < 1000:
            logger.info("Skipping retraining: insufficient recent flows in window.")
            return
            
        recent_x = np.array([f['features'] for f in self.all_recent_flows])
        recent_y = np.array([f['label'] for f in self.all_recent_flows])
        
        # 1. Update K-Means (only on recent benign flows)
        benign_indices = [i for i, f in enumerate(self.all_recent_flows) if f['is_benign']]
        if len(benign_indices) >= 100:
            recent_benign_x = recent_x[benign_indices]
            self.spatial_filter.partial_fit(recent_benign_x)
            self.spatial_filter.save(self.config['models']['kmeans']['save_path'])
        
        # 2. Update GRU/LSTM
        candidate_mask = self.spatial_filter.predict_anomalies(recent_x)
        recent_cand_x = recent_x[candidate_mask]
        recent_cand_y = recent_y[candidate_mask]
        
        ben_cand = np.where(recent_cand_y == 0)[0]
        att_cand = np.where(recent_cand_y == 1)[0]
        
        # Balance candidates
        if len(ben_cand) > 0 and len(att_cand) > 0:
            if len(att_cand) > len(ben_cand):
                np.random.shuffle(att_cand)
                keep_att = att_cand[:len(ben_cand)]
                balanced_idx = np.concatenate([ben_cand, keep_att])
            else:
                np.random.shuffle(ben_cand)
                keep_ben = ben_cand[:len(att_cand)]
                balanced_idx = np.concatenate([keep_ben, att_cand])
            np.random.shuffle(balanced_idx)
            recent_cand_x = recent_cand_x[balanced_idx]
            recent_cand_y = recent_cand_y[balanced_idx]
        
        if len(recent_cand_x) >= self.seq_length:
            self.temporal_classifier.fine_tune(recent_cand_x, recent_cand_y, epochs=1, batch_size=32)
            self.temporal_classifier.save(self.config['models']['lstm']['save_path'])
            
        # 3. Update Drift Detector Reference
        if len(benign_indices) >= 500:
            ref_size = min(len(benign_indices), self.config['drift']['reference_size'])
            indices = np.random.choice(benign_indices, ref_size, replace=False)
            new_ref = recent_x[indices]
            self.drift_detector.update_reference(new_ref)
            
        self.drift_detector.clear_live_window()
        logger.info("Model retraining and drift detector adaptation complete.")

class StreamFileHandler(FileSystemEventHandler):
    def __init__(self, consumer: UnifiedStreamingConsumer):
        self.consumer = consumer
        self.processed_files = set()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".json"):
            return
        if event.src_path in self.processed_files:
            return
        
        time.sleep(0.05)
        logger.info(f"New stream file detected: {os.path.basename(event.src_path)}")
        try:
            with open(event.src_path, 'r') as f:
                batch_data = json.load(f)
            self.consumer.process_batch(batch_data)
            self.processed_files.add(event.src_path)
            os.remove(event.src_path)
        except Exception as e:
            logger.error(f"Error processing stream file {event.src_path}: {e}")

def main():
    consumer = UnifiedStreamingConsumer()
    os.makedirs(consumer.local_stream_dir, exist_ok=True)
    event_handler = StreamFileHandler(consumer)
    observer = Observer()
    observer.schedule(event_handler, path=consumer.local_stream_dir, recursive=False)
    observer.start()
    
    logger.info(f"Streaming consumer active. Monitoring: {consumer.local_stream_dir}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
