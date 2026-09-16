import os
import pickle
import logging
import json
import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, window, struct
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType, BooleanType

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Schema definition for incoming Kafka messages
schema = StructType([
    StructField("duration", DoubleType(), True),
    StructField("src_packets", DoubleType(), True),
    StructField("dst_packets", DoubleType(), True),
    StructField("src_bytes", DoubleType(), True),
    StructField("dst_bytes", DoubleType(), True),
    StructField("rate", DoubleType(), True),
    StructField("src_packet_mean", DoubleType(), True),
    StructField("dst_packet_mean", DoubleType(), True),
    StructField("src_win_bytes", DoubleType(), True),
    StructField("dst_win_bytes", DoubleType(), True),
    StructField("label", IntegerType(), True),
    StructField("attack_cat", StringType(), True),
    StructField("ingest_time", LongType(), True)
])

# Spark UDFs for Stage 1 (K-Means) and Stage 2 (LSTM) Anomaly Detection
# Note: Models are loaded inside the worker context (lazy loading) to avoid serialization errors

spatial_filter_broadcast = None
temporal_classifier_broadcast = None
scaler_broadcast = None

def load_models_local():
    global spatial_filter_broadcast, temporal_classifier_broadcast, scaler_broadcast
    
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    
    if scaler_broadcast is None:
        scaler_path = os.path.join(processed_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler_broadcast = pickle.load(f)
            
    if spatial_filter_broadcast is None:
        from src.models.kmeans import SpatialAnomalyFilter
        kmeans_path = os.path.join(processed_dir, "kmeans_model.pkl")
        spatial_filter_broadcast = SpatialAnomalyFilter.load(kmeans_path)
        
    if temporal_classifier_broadcast is None:
        from src.models.lstm import LSTMTemporalClassifier
        lstm_path = os.path.join(processed_dir, "lstm_model.pt")
        temporal_classifier_broadcast = LSTMTemporalClassifier.load(lstm_path)

@udf(returnType=BooleanType())
def run_stage1_filter(duration, src_packets, dst_packets, src_bytes, dst_bytes, rate, 
                       src_packet_mean, dst_packet_mean, src_win_bytes, dst_win_bytes):
    """Stage 1 K-Means filter UDF. Flags candidate outliers."""
    load_models_local()
    
    features = np.array([[duration, src_packets, dst_packets, src_bytes, dst_bytes, 
                          rate, src_packet_mean, dst_packet_mean, src_win_bytes, dst_win_bytes]])
    processed_features = np.log1p(np.maximum(0, features))
    scaled_features = scaler_broadcast.transform(processed_features)
    
    is_candidate = spatial_filter_broadcast.predict_anomalies(scaled_features)[0]
    return bool(is_candidate)

@udf(returnType=DoubleType())
def run_stage2_classifier(duration, src_packets, dst_packets, src_bytes, dst_bytes, rate, 
                            src_packet_mean, dst_packet_mean, src_win_bytes, dst_win_bytes):
    """Stage 2 GRU prediction UDF."""
    load_models_local()
    
    features = np.array([[duration, src_packets, dst_packets, src_bytes, dst_bytes, 
                          rate, src_packet_mean, dst_packet_mean, src_win_bytes, dst_win_bytes]])
    processed_features = np.log1p(np.maximum(0, features))
    scaled_features = scaler_broadcast.transform(processed_features)
    
    # Simulate a temporal sequence by padding the current sample to sequence length
    seq_len = temporal_classifier_broadcast.seq_length
    padded_seq = np.repeat(scaled_features, seq_len, axis=0) # Padded sequence
    
    prob = temporal_classifier_broadcast.predict_single_flow(padded_seq)
    return float(prob)

def main():
    spark = SparkSession.builder \
        .appName("HybridIntrusionDetectionPipeline") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark Session successfully initialized.")
    
    bootstrap_servers = "localhost:9092"
    kafka_topic = "network-traffic"
    
    # Read stream from Kafka
    logger.info(f"Subscribing to Kafka topic: {kafka_topic}...")
    df_kafka = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "latest") \
        .load()
        
    # Deserialize JSON value
    df_parsed = df_kafka.selectExpr("CAST(value AS STRING) as json_str") \
        .select(col("json_str")) \
        .select(udf(lambda x: json.loads(x), schema)("json_str").alias("data")) \
        .select("data.*")
        
    # Apply Stage 1: Spatial Filter (Mini-batch K-Means)
    df_filtered = df_parsed.withColumn(
        "is_candidate", 
        run_stage1_filter(
            "duration", "src_packets", "dst_packets", "src_bytes", "dst_bytes",
            "rate", "src_packet_mean", "dst_packet_mean", "src_win_bytes", "dst_win_bytes"
        )
    )
    
    # Apply Stage 2: Temporal LSTM Classifier (for candidate anomalies only)
    df_processed = df_filtered.withColumn(
        "anomaly_prob",
        col("is_candidate").cast("integer") * run_stage2_classifier(
            "duration", "src_packets", "dst_packets", "src_bytes", "dst_bytes",
            "rate", "src_packet_mean", "dst_packet_mean", "src_win_bytes", "dst_win_bytes"
        )
    ).withColumn(
        "prediction",
        col("anomaly_prob") > 0.5
    )
    
    # Output streaming predictions to Console / Sink
    logger.info("Starting Streaming Query...")
    query = df_processed.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()
        
    query.awaitTermination()

if __name__ == "__main__":
    main()
