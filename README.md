# MLP Baseline & Data Ingestion Pipeline — Network Intrusion Detection

**Project Title:** Hybrid Deep Learning Approaches for Network Intrusion Detection: A Comparative Study  
**Course:** Deep Learning Project [ICT-4442]  
**Contributor:** Simran Singhania (Reg. No: 230911126)  
**Assigned Model:** Multi-Layer Perceptron (MLP Baseline) & Data Ingestion Pipeline  

---

## 📌 Executive Summary

This repository module contains the implementation of the **MLP (Multi-Layer Perceptron) Baseline Neural Network** and the **End-to-End Data Preprocessing & Streaming Ingestion Pipeline** for binary network intrusion detection (benign vs. attack traffic).

As part of the comparative deep learning study across feedforward, convolutional (1D-CNN), and recurrent (BiLSTM) architectures, this module establishes:
1. A **leak-free preprocessing pipeline** mapping raw CICIDS2017 (~2.8M flows) and UNSW-NB15 (~257K flows) datasets to a unified 10-feature flow-level schema.
2. An **ultra-high throughput MLP baseline classifier** implemented in PyTorch.
3. A **simulated real-time streaming traffic producer** supporting both local directory JSON buffering and Apache Kafka replay.
4. Comprehensive **per-attack category error analysis and performance logging**.

---

## 🏗️ Architecture & Implementation Details

### 1. Unified 10-Feature Schema & Data Preprocessing (`src/ingestion/preprocess.py`)
- **Unified Features (10):** `duration`, `src_packets`, `dst_packets`, `src_bytes`, `dst_bytes`, `rate`, `src_packet_mean`, `dst_packet_mean`, `src_win_bytes`, `dst_win_bytes`.
- **Cleaning & Imputation:** Infinite values (`inf`, `-inf`) are replaced with `NaN` and median-imputed.
- **Logarithmic Compression:** Applies $x' = \log(1 + \max(0, x))$ (`log1p`) to compress extreme positive skew across traffic byte volumes and transmission rates spanning 7 orders of magnitude.
- **Leak-Free Normalization:** Fits `StandardScaler` **exclusively on benign CICIDS2017 training flows**, freezing the scaler parameters ($\mu_{\text{benign}}, \sigma_{\text{benign}}$) across all evaluation splits and zero-shot cross-dataset testing on UNSW-NB15 to prevent data leakage.
- **Stratified Splitting & Class Balancing:** 80/20 train-test stratified split with 1:1 class balancing during training.

### 2. MLP Baseline Architecture (`src/models/mlp.py`)
A feedforward deep neural network processing each flow independently without sequential or spatial induction biases:
- **Network Topology:** `Input(10) → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(64) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(32) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(1) → Sigmoid`
- **Loss Function:** Binary Cross-Entropy (`BCELoss`)
- **Optimization:** Adam Optimizer ($\text{lr} = 1\times 10^{-3}$, weight decay $= 1\times 10^{-5}$) with `ReduceLROnPlateau` learning rate scheduling.

### 3. Traffic Replay Producer (`src/ingestion/producer.py`)
- Simulates real-time network flow streaming by replaying parquet dataset records in sub-second batches.
- Supports dual streaming modes: `local` (writing micro-batches as JSON files) and `kafka` (publishing to an Apache Kafka topic).

---

## 📊 Benchmark Results (CICIDS2017 Test Set — 50,000 Samples)

| Metric | MLP Baseline Result | Notes / Highlight |
| :--- | :--- | :--- |
| **Accuracy** | **99.08%** | High baseline detection accuracy |
| **Precision** | **95.29%** | Low false positive rate on benign traffic |
| **Recall** | **98.48%** | High detection rate on malicious flows |
| **F1-Score** | **0.9686** | Balanced binary classification performance |
| **ROC-AUC** | **0.9994** | Outstanding overall class separability |
| **Batch Time (50k)** | **0.021 s** | Near-instant batch inference |
| **Throughput** | **2,357,622 flows/sec** | **Ultra-high throughput classical DL baseline** |

---

## 📁 Directory & File Structure

```text
.
├── config.yaml                    # Master project configuration (features, paths, hyperparams)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules for venv, caches, and stream buffers
├── README.md                      # Module documentation
├── data/
│   ├── scaler.pkl                 # Fitted leak-free StandardScaler artifact
│   ├── mlp_model.pt               # Trained PyTorch MLP model checkpoint
│   └── error_analysis_mlp.txt     # Detailed per-attack category error analysis report
├── src/
│   ├── ingestion/
│   │   ├── preprocess.py          # Data cleaning, log1p transform, scaler fitting & dataset export
│   │   └── producer.py            # Simulated streaming traffic replay (Local/Kafka)
│   └── models/
│       └── mlp.py                 # PyTorch MLPNetwork & MLPClassifier wrapper
└── tests/
    └── test_preprocess.py         # Unit tests for preprocessing & label normalization
```

---

## 🚀 How to Run & Verify

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
pytest tests
```

### 3. Run Preprocessing Pipeline
```bash
python src/ingestion/preprocess.py
```

### 4. Run Streaming Producer Simulation
```bash
python src/ingestion/producer.py --rate 500 --max-flows 5000
```

---

## ✍️ Author & Responsibilities

- **Name:** Simran Singhania  
- **Reg. No.:** 230911126  
- **Responsibilities:** 
  - Design, training, and tuning of MLP Baseline (`mlp.py`).
  - End-to-end data ingestion & preprocessing pipeline (`preprocess.py`).
  - Leak-free `StandardScaler` & `log1p` feature compression.
  - Throughput benchmarking & per-attack error analysis writeup.
