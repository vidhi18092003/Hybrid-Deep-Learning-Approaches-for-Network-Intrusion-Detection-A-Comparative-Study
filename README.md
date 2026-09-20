# Hybrid Deep Learning Approaches for Network Intrusion Detection: A Comparative Study

**ICT-4442 Deep Learning Mini-Project**  
*School of Computer Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education*  

- **Team Leader:** Vidhi Ajmera (Registration No: `230911124`)
- **Contributor:** Aditya Raj Sharma (Registration No: `230953410`)
- **Aditya's Institutional Email:** `aditya.sharma@learner.manipal.edu`
- **Aditya's Assigned Module:** 1D-CNN architecture, multi-model training, comparative evaluation, benchmark visualization, and Streamlit dashboard

---

## 📌 Executive Summary

Modern enterprise networks generate high-velocity streaming traffic that conventional signature-based Intrusion Detection Systems (IDS) and static batch-trained machine learning models fail to protect against due to zero-day attack vectors and concept drift. 

This repository implements a **production-grade, drift-aware hybrid intrusion detection system** combining:
1. **Unsupervised Spatial Outlier Filtering** (Mini-Batch K-Means) to rapidly eliminate benign traffic overhead.
2. **Deep Sequential Modeling via Bidirectional LSTM (BiLSTM)** with a sliding window ($W=5$) to capture forward and backward temporal dependencies across consecutive network flows.
3. **Statistical Concept Drift Detection Engine** using the two-sample Kolmogorov-Smirnov (KS-test) on rolling feature distributions to detect network traffic distribution shifts.
4. **Streaming Architecture** supporting standalone real-time flow queues as well as distributed PySpark Structured Streaming and Apache Kafka pipelines.
5. **1D Convolutional Classification** using three Conv1D stages to learn local interactions across the unified network-flow features.
6. **Comparative Evaluation & Visualization** for MLP, 1D-CNN, and BiLSTM models on CICIDS2017 and UNSW-NB15, with confusion matrices, ROC curves, metric tables, and error analysis.
7. **Interactive Monitoring Dashboard** built with Streamlit for live alerts and offline benchmark exploration.
8. **Containerized Deployment** using multi-stage Docker and Docker Compose orchestration.

---

## 🏗️ System Architecture

```text
               ┌──────────────────────────────────────────────────┐
               │    Streaming Network Flows (CICIDS2017 / UNSW)   │
               └────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │   Unified 10-Feature Schema & Robust Normalizer  │
               └────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │   STAGE 1: Spatial Outlier Filter (Mini-Batch)   │
               │   Centroid Distance Thresholding (K=16 Clusters) │
               └────────────────────────┬─────────────────────────┘
                                        │
                            [Outlier Flagged: True]
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │   STAGE 2: Bidirectional LSTM Temporal Network   │
               │  Sliding Window W=5, Hidden=64, BCE Anomaly Prob │
               └────────────────────────┬─────────────────────────┘
                                        │
                         [Confirmed Intrusion Detected]
                                        │
                                        ▼
               ┌──────────────────────────────────────────────────┐
               │   Kolmogorov-Smirnov (KS-Test) Drift Monitor     │
               │  Rolling Windows (N=1000, alpha=0.05, Trigger>=3)│
               └──────────────────────────────────────────────────┘
```

---

## 📊 Experimental Results & Model Evaluation

The models were evaluated under rigorous conditions on standardized network flow benchmarks with balanced validation and test splits:

| Architecture | Inductive Bias / Family | Parameters | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **BiLSTM (Primary)** | **Bidirectional Recurrent ($W=5$)** | **77,313** | **99.16%** | **95.91%** | **98.40%** | **97.14%** | **0.9992** |
| 1D-CNN | Three-stage convolutional network | 39,809 | 99.00% | 95.28% | 97.93% | 96.59% | 0.9990 |
| LSTM (Ablation) | Unidirectional Recurrent ($W=5$) | 38,913 | 98.62% | 94.10% | 96.50% | 95.28% | 0.9961 |
| Mini-Batch K-Means | Unsupervised Centroid Clustering | K=16 | 93.40% | 88.20% | 91.10% | 89.62% | 0.9450 |

### Key Findings:
- **Temporal Context Advantage:** The Bidirectional LSTM achieved the highest overall F1-score (**97.14%**) and ROC-AUC (**0.9992**), proving that network intrusions manifest sequentially across successive packets/flows (e.g., reconnaissance port scans followed by vulnerability exploitation).
- **1D-CNN Baseline:** The three-stage 1D-CNN achieved **99.00% accuracy**, **96.59% F1-score**, and **0.9990 ROC-AUC** on CICIDS2017 while modeling local feature interactions.
- **Reproducible Comparison:** The evaluation suite generates model comparison tables, ROC curves, confusion matrices, metric bar charts, and model-specific error reports for CICIDS2017 and UNSW-NB15.
- **Ablation Insight:** Bidirectional temporal modeling outperforms unidirectional LSTM by $+1.86\%$ in F1-score, confirming the benefit of both past and future flow context within the 5-step sequence window.
- **Drift Robustness:** The Kolmogorov-Smirnov drift module reliably identifies distribution divergence across statistical traffic shifts, triggering proactive pipeline alerts.

---

## 📁 Repository Structure

```text
├── src/
│   ├── models/
│   │   ├── bilstm.py             # Primary 2-layer BiLSTM model architecture
│   │   ├── cnn.py                # Three-stage 1D-CNN model and classifier wrapper
│   │   ├── lstm.py               # Comparative Unidirectional LSTM ablation model
│   │   ├── kmeans.py             # Mini-Batch K-Means spatial outlier gating filter
│   │   ├── train_all_models.py   # Shared MLP, 1D-CNN, and BiLSTM training workflow
│   │   └── train_initial.py      # Initial hybrid pipeline training workflow
│   ├── evaluation/
│   │   ├── compare_models.py     # Comparative metrics, plots, and error analysis
│   │   └── evaluator.py          # CICIDS2017/UNSW-NB15 pipeline evaluation harness
│   ├── drift/
│   │   └── detector.py           # Two-sample KS-test statistical concept drift detector
│   └── streaming/
│       ├── consumer.py           # Real-time event queue streaming consumer
│       └── spark_consumer.py     # Production PySpark Structured Streaming consumer
├── dashboard/
│   └── app.py                    # Streamlit live monitor and benchmark dashboard
├── tests/
│   └── test_drift.py             # Comprehensive test suite for drift detection
├── data/
│   ├── bilstm_model.pt           # Trained PyTorch weights for BiLSTM model
│   ├── cnn_model.pt              # Trained PyTorch weights for 1D-CNN model
│   ├── lstm_model.pt             # Trained PyTorch weights for LSTM baseline
│   ├── kmeans_model.pkl          # Serialized Mini-Batch K-Means cluster centroids
│   ├── comparison_*.csv          # CICIDS2017 and UNSW-NB15 comparison tables
│   ├── comparison_*.png          # ROC, confusion-matrix, and metric visualizations
│   └── error_analysis_1d_cnn.txt # 1D-CNN prediction error analysis
├── config.yaml                   # Global hyperparameters, schemas, and pipeline settings
├── pyproject.toml                # Project metadata, build settings, and test configuration
├── Dockerfile                    # Multi-stage production container specification
├── docker-compose.yml            # Multi-service stack orchestration (Kafka, Spark, IDS)
├── run_demo.ps1                  # PowerShell automated demo execution script
└── README.md                     # Project documentation & architectural specification
```

---

## ⚙️ Environment Setup & Quick Start

### 1. Requirements & Prerequisites
- Python 3.10+
- PyTorch 2.0+
- PySpark 3.4+ / Scikit-Learn / NumPy / Pandas / PyYAML
- Streamlit / Matplotlib / Seaborn

### 2. Running Automated Unit Tests
Verify model initialization, tensor shapes, and the concept drift detection logic:
```bash
pytest tests/
```

### 3. Running the Pipeline Demo (PowerShell)
Execute the end-to-end verification script:
```powershell
.\run_demo.ps1
```

### 4. Running Distributed Container Stack
Launch Kafka, Zookeeper, and streaming services via Docker Compose:
```bash
docker-compose up -d
```

### 5. Running Aditya's Evaluation and Dashboard
After the datasets and companion team modules are available, generate the comparison artifacts and open the dashboard:
```bash
python -m src.evaluation.compare_models
streamlit run dashboard/app.py
```

---

## 🎓 Academic Integrity & Course Declaration
This repository contains the team codebase for the **ICT-4442 Deep Learning Mini-Project**, Department of Computer Science & Engineering, Manipal Institute of Technology. The 1D-CNN, multi-model training, comparative evaluation, benchmark artifacts, and Streamlit dashboard on `aditya_branch` were contributed by **Aditya Raj Sharma** (Reg. No. `230953410`).
