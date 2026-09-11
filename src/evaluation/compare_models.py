"""
Comparative Model Evaluation Script.
Evaluates all 3 deep learning models (MLP, 1D-CNN, BiLSTM) on the same test sets
using the same metrics, and generates:
  - A single comparison table (printed + CSV)
  - Per-model confusion matrices
  - ROC curve overlay plot
  - Error analysis for the worst-performing model
"""

import os
import sys
import pickle
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.ingestion.preprocess import load_config
from src.models.mlp import MLPClassifier
from src.models.cnn import CNN1DClassifier
from src.models.bilstm import BiLSTMClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data(config: dict, dataset_name: str):
    """Load test data and apply preprocessing."""
    processed_dir = config['data']['processed_dir']
    features = config['data']['features']
    log_transform = config['data'].get('log_transform', False)

    if dataset_name == "cicids2017":
        path = os.path.join(processed_dir, "cicids2017_test.parquet")
    else:
        path = os.path.join(processed_dir, "unsw_nb15_test.parquet")

    if not os.path.exists(path):
        logger.error(f"Test data not found: {path}")
        return None, None

    df = pd.read_parquet(path)

    # Subsample for tractable evaluation (especially for BiLSTM sequential inference)
    if len(df) > 50000:
        df = df.sample(50000, random_state=42).sort_index()

    X_raw = df[features].values
    if log_transform:
        X_processed = np.log1p(np.maximum(0, X_raw))
    else:
        X_processed = X_raw

    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X_processed)
    y_true = df['label'].values

    return X_scaled, y_true


def evaluate_model(model, model_name: str, X: np.ndarray, y_true: np.ndarray) -> dict:
    """Evaluate a single model and return metrics."""
    logger.info(f"  Evaluating {model_name}...")

    start_time = time.time()
    y_proba = model.predict_proba(X)
    y_pred = (y_proba >= 0.5).astype(int)
    inference_time = time.time() - start_time

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.5

    throughput = len(X) / max(inference_time, 0.001)

    logger.info(f"    Acc={acc:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'Inference Time (s)': round(inference_time, 2),
        'Throughput (flows/s)': round(throughput, 0),
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def plot_confusion_matrices(results: list, y_true: np.ndarray, dataset_name: str, output_dir: str):
    """Plot confusion matrices for all models side by side."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_true, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'], ax=ax)
        ax.set_title(f"{res['Model']}\nF1={res['F1-Score']:.4f}", fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.suptitle(f'Confusion Matrices — {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"comparison_confusion_{dataset_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrices to {filepath}")


def plot_roc_curves(results: list, y_true: np.ndarray, dataset_name: str, output_dir: str):
    """Plot ROC curves for all models on one figure."""
    plt.figure(figsize=(8, 6))

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_true, res['y_proba'])
        plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                 label=f"{res['Model']} (AUC={res['ROC-AUC']:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves — {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"comparison_roc_{dataset_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve plot to {filepath}")


def plot_metric_bars(comparison_df: pd.DataFrame, dataset_name: str, output_dir: str):
    """Plot bar chart comparing models across key metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_plot = comparison_df.set_index('Model')[metrics]

    ax = df_plot.plot(kind='bar', figsize=(10, 6), width=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison — {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"comparison_bars_{dataset_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Saved metric bar chart to {filepath}")


def error_analysis(model, model_name: str, X: np.ndarray, y_true: np.ndarray, output_dir: str):
    """Perform error analysis for a model — identify failure patterns."""
    y_proba = model.predict_proba(X)
    y_pred = (y_proba >= 0.5).astype(int)

    false_positives = np.where((y_pred == 1) & (y_true == 0))[0]
    false_negatives = np.where((y_pred == 0) & (y_true == 1))[0]

    logger.info(f"\n--- Error Analysis for {model_name} ---")
    logger.info(f"  Total test samples: {len(y_true)}")
    logger.info(f"  False Positives (benign flagged as attack): {len(false_positives)}")
    logger.info(f"  False Negatives (attack missed): {len(false_negatives)}")

    if len(false_positives) > 0:
        fp_proba = y_proba[false_positives]
        logger.info(f"  FP confidence — Mean: {np.mean(fp_proba):.4f}, Std: {np.std(fp_proba):.4f}")

    if len(false_negatives) > 0:
        fn_proba = y_proba[false_negatives]
        logger.info(f"  FN confidence — Mean: {np.mean(fn_proba):.4f}, Std: {np.std(fn_proba):.4f}")

    # Write error analysis to file
    analysis_path = os.path.join(output_dir, f"error_analysis_{model_name.lower().replace(' ', '_').replace('-', '_')}.txt")
    with open(analysis_path, 'w') as f:
        f.write(f"Error Analysis: {model_name}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Correct predictions: {np.sum(y_pred == y_true)}\n")
        f.write(f"False Positives: {len(false_positives)}\n")
        f.write(f"False Negatives: {len(false_negatives)}\n\n")
        f.write(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=['Benign', 'Attack'])}\n")
    logger.info(f"  Error analysis saved to {analysis_path}")


def main():
    config = load_config()
    processed_dir = config['data']['processed_dir']
    output_dir = processed_dir  # Save plots alongside data

    # ── Load Models ──
    logger.info("Loading trained models...")

    mlp_path = config.get('models', {}).get('mlp', {}).get('save_path', os.path.join(processed_dir, "mlp_model.pt"))
    cnn_path = config.get('models', {}).get('cnn', {}).get('save_path', os.path.join(processed_dir, "cnn_model.pt"))
    bilstm_path = config.get('models', {}).get('bilstm', {}).get('save_path', os.path.join(processed_dir, "bilstm_model.pt"))

    models = {}
    if os.path.exists(mlp_path):
        models['MLP'] = MLPClassifier.load(mlp_path)
    else:
        logger.warning(f"MLP model not found at {mlp_path}. Skipping.")

    if os.path.exists(cnn_path):
        models['1D-CNN'] = CNN1DClassifier.load(cnn_path)
    else:
        logger.warning(f"1D-CNN model not found at {cnn_path}. Skipping.")

    if os.path.exists(bilstm_path):
        models['BiLSTM'] = BiLSTMClassifier.load(bilstm_path)
    else:
        logger.warning(f"BiLSTM model not found at {bilstm_path}. Skipping.")

    if not models:
        logger.error("No models found. Please run train_all_models.py first.")
        return

    # ── Evaluate on CICIDS2017 Test Set ──
    for dataset_name, dataset_key in [("CICIDS2017", "cicids2017"), ("UNSW-NB15", "unsw_nb15")]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EVALUATING ALL MODELS ON {dataset_name}")
        logger.info(f"{'=' * 60}")

        X_test, y_test = load_and_prepare_data(config, dataset_key)
        if X_test is None:
            continue

        logger.info(f"Test set: {len(y_test)} samples (Benign={np.sum(y_test == 0)}, Attack={np.sum(y_test == 1)})")

        results = []
        for model_name, model in models.items():
            res = evaluate_model(model, model_name, X_test, y_test)
            results.append(res)

        # ── Comparison Table ──
        display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC',
                        'Inference Time (s)', 'Throughput (flows/s)']
        comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k in display_cols} for r in results])

        # Round numeric columns for display
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            comparison_df[col] = comparison_df[col].round(4)

        logger.info(f"\n{'─' * 80}")
        logger.info(f"COMPARISON TABLE — {dataset_name}")
        logger.info(f"{'─' * 80}")
        print(comparison_df.to_string(index=False))
        print()

        # Save comparison table
        csv_path = os.path.join(output_dir, f"comparison_table_{dataset_key}.csv")
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Comparison table saved to {csv_path}")

        # ── Plots ──
        plot_confusion_matrices(results, y_test, dataset_name, output_dir)
        plot_roc_curves(results, y_test, dataset_name, output_dir)
        plot_metric_bars(comparison_df, dataset_name, output_dir)

        # ── Error Analysis (for worst-performing model by F1) ──
        worst = min(results, key=lambda r: r['F1-Score'])
        error_analysis(models[worst['Model']], worst['Model'], X_test, y_test, output_dir)

    logger.info(f"\n{'=' * 60}")
    logger.info("COMPARATIVE EVALUATION COMPLETE")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
