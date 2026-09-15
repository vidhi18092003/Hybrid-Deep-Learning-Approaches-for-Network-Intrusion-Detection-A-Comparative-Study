import os
import sys
import time
import json
import logging
import yaml
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from kafka import KafkaProducer
from src.ingestion.preprocess import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficStreamProducer:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.stream_mode = self.config['streaming']['stream_mode']
        self.features = self.config['data']['features']
        
        # Setup target directory for local mode
        self.local_dir = self.config['streaming']['local_stream_dir']
        if self.stream_mode == "local":
            os.makedirs(self.local_dir, exist_ok=True)
            logger.info(f"Local streaming directory set to: {self.local_dir}")
            # Clear old files in local stream dir to start fresh
            for f in os.listdir(self.local_dir):
                if f.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.local_dir, f))
                    except Exception as e:
                        logger.warning(f"Could not remove old stream file {f}: {e}")

        # Setup Kafka Producer if in kafka mode
        self.producer = None
        if self.stream_mode == "kafka":
            bootstrap_servers = self.config['streaming']['kafka_bootstrap_servers']
            logger.info(f"Connecting to Kafka bootstrap servers at: {bootstrap_servers}")
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',
                    retries=3
                )
                logger.info("Kafka Producer successfully initialized.")
            except Exception as e:
                logger.error(f"Failed to connect to Kafka: {e}. Falling back to 'local' directory streaming.")
                self.stream_mode = "local"
                os.makedirs(self.local_dir, exist_ok=True)

    def run_replay(self, dataset_path: str, rate: int = 100, max_flows: int = None, loop: bool = False):
        """
        Replays network traffic from a Parquet dataset.
        
        Args:
            dataset_path: str path to the parquet file to stream.
            rate: int target flows per second.
            max_flows: int optional max number of flows to replay per pass.
            loop: bool whether to loop indefinitely.
        """
        logger.info(f"Loading replay dataset: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Replay dataset not found at {dataset_path}")
            
        df = pd.read_parquet(dataset_path)
        total_rows = len(df)
        if max_flows:
            df = df.head(max_flows)
            total_rows = len(df)
            
        logger.info(f"Starting replay of {total_rows} flows at {rate} flows/sec in '{self.stream_mode}' mode (loop={loop}).")
        
        batch_size = max(1, rate // 5) # Write in sub-second batches (e.g. 5 times per second)
        sleep_interval = batch_size / rate
        
        kafka_topic = self.config['streaming']['kafka_topic']
        records = df.to_dict(orient='records')
        
        batch_idx = 0
        total_streamed = 0
        
        while True:
            idx = 0
            while idx < total_rows:
                batch = records[idx : idx + batch_size]
                idx += batch_size
                total_streamed += len(batch)
                
                timestamp = int(time.time() * 1000)
                
                # Enrich records with a production ingestion timestamp
                for record in batch:
                    record['ingest_time'] = timestamp
                    # Convert numpy types to native types for JSON serialization
                    for k, v in record.items():
                        if hasattr(v, 'item'): # numpy types
                            record[k] = v.item()
                
                if self.stream_mode == "kafka":
                    try:
                        for record in batch:
                            self.producer.send(kafka_topic, value=record)
                        self.producer.flush()
                    except Exception as e:
                        logger.error(f"Error sending to Kafka: {e}")
                else:
                    # Local directory streaming: write batch to a JSON file
                    batch_file = os.path.join(self.local_dir, f"batch_{batch_idx:06d}_{timestamp}.json")
                    try:
                        with open(batch_file, 'w') as f:
                            # Write as a JSON array
                            json.dump(batch, f)
                        batch_idx += 1
                    except Exception as e:
                        logger.error(f"Error writing local streaming batch: {e}")
                
                if total_streamed % (rate * 5) == 0 or idx >= total_rows:
                    logger.info(f"Streamed {total_streamed} total flows (pass flow {min(idx, total_rows)}/{total_rows}).")
                    
                time.sleep(sleep_interval)
            
            if not loop:
                break
            logger.info("Dataset end reached. Looping stream replay back to start...")
            
        logger.info("Stream replay completed.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simulated Network Traffic Streaming Producer")
    parser.add_argument("--dataset", type=str, default="cicids2017_test.parquet", 
                        help="Dataset file name in the processed directory (e.g. cicids2017_test.parquet or unsw_nb15_test.parquet)")
    parser.add_argument("--rate", type=int, default=None, help="Flows per second")
    parser.add_argument("--max-flows", type=int, default=None, help="Max flows to stream per pass")
    parser.add_argument("--loop", action="store_true", help="Continuously loop stream replay indefinitely")
    args = parser.parse_args()
    
    config = load_config()
    processed_dir = config['data']['processed_dir']
    dataset_path = os.path.join(processed_dir, args.dataset)
    
    stream_rate = args.rate if args.rate else config['streaming']['local_stream_speed']
    
    producer = TrafficStreamProducer()
    producer.run_replay(dataset_path, rate=stream_rate, max_flows=args.max_flows, loop=args.loop)

if __name__ == "__main__":
    main()
