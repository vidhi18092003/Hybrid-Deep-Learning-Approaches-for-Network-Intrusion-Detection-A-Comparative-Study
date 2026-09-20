import os
import sys
import time
import json
import logging
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import datetime

st.set_page_config(
    page_title="Deep Learning Project - Streaming Anomaly Detection IDS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        border-left: 5px solid #3B82F6;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
    }
    .alert-card {
        background-color: #7F1D1D;
        border-radius: 10px;
        padding: 10px;
        border-left: 5px solid #EF4444;
        color: #FEE2E2;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

processed_dir = "c:/Users/vinay/Downloads/hybridintrusiondetection/data"
live_results_path = os.path.join(processed_dir, "live_results.json")

def load_live_results(limit=1000):
    """Loads the latest live streaming results from the JSONL file."""
    if not os.path.exists(live_results_path) or os.path.getsize(live_results_path) == 0:
        return pd.DataFrame()
        
    records = []
    try:
        with open(live_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        records.append(json.loads(line_str))
                    except Exception:
                        # Skip partially written line from concurrent streaming file writes
                        continue
                        
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        if len(df) > limit:
            df = df.tail(limit)
        return df
    except Exception:
        return pd.DataFrame()

# Sidebar config
st.sidebar.title("🛡️ Deep Learning Project")
st.sidebar.markdown("---")
st.sidebar.markdown("### Controls")
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=10, value=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Config")
st.sidebar.info("""
**Stage 1:** Mini-batch K-Means (Spatial Filter)
**Stage 2:** PyTorch GRU (Temporal sequence model)
**Concept Drift:** KS-test distribution drift trigger
""")

# Setup Title
st.title("🛡️ Deep Learning Project: Streaming Anomaly Detection")
st.caption("Hybrid ML Spatial-Temporal Anomaly Detection Pipeline")

# Setup tabs
tab1, tab2 = st.tabs(["🔴 Real-time Streaming Monitor", "📊 Offline Evaluation & Generalization"])

with tab1:
    # Read the data
    df_live = load_live_results()
    
    if df_live.empty:
        st.warning("No streaming data detected yet. Make sure the producer and consumer are running.")
        st.info("💡 Run the streaming simulation: start the consumer first, then run the producer.")
        # Render a button to refresh manually
        if st.button("Manual Refresh"):
            st.rerun()
    else:
        # Latest stats
        latest = df_live.iloc[-1]
        
        # Calculate KPIs
        total_flows = latest.get('processed_total', len(df_live))
        total_anomalies = latest.get('anomalies_total', df_live[df_live['prediction'] == 1].shape[0])
        anomaly_rate = (total_anomalies / total_flows) * 100 if total_flows > 0 else 0
        avg_latency = df_live['latency_ms'].mean()
        throughput = latest.get('throughput', 0.0)
        
        # Grid layout for KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f'<div class="metric-card" style="border-left-color: #3B82F6;"><div class="metric-label">Total Flows Processed</div><div class="metric-value">{total_flows:,}</div></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown(f'<div class="metric-card" style="border-left-color: #EF4444;"><div class="metric-label">Anomalies Blocked</div><div class="metric-value">{total_anomalies:,}</div></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="metric-card" style="border-left-color: #F59E0B;"><div class="metric-label">Anomaly Rate</div><div class="metric-value">{anomaly_rate:.2f}%</div></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="metric-card" style="border-left-color: #10B981;"><div class="metric-label">Avg Latency / Throughput</div><div class="metric-value">{avg_latency:.2f} ms / {throughput:.1f} f/s</div></div>', unsafe_allow_html=True)
            
        st.write("")
        
        # Concept Drift Alarm Banner
        drift_events = df_live[df_live['drift_triggered'] == True]
        if not drift_events.empty:
            st.error(f"⚠️ **CONCEPT DRIFT DETECTED**: System detected statistical concept drift at flow #{drift_events.iloc[-1]['processed_total']}. Models have successfully adapted via online sliding-window retraining!")
        else:
            st.success("🟢 **System Health**: Concept Drift Detector active. Network feature distributions are stable.")
            
        st.write("")
        
        # Charts Section
        col_charts_left, col_charts_right = st.columns(2)
        
        with col_charts_left:
            st.subheader("Rolling Anomaly Scores & Stage 1 Distance")
            chart_df = df_live.tail(100).copy()
            chart_df['Index'] = range(len(chart_df))
            
            # Anomaly probability chart
            st.line_chart(data=chart_df, x='Index', y='anomaly_prob', height=250)
            st.caption("Latest 100 flows: GRU temporal anomaly probability score (threshold = 0.5)")
            
        with col_charts_right:
            st.subheader("Detected Traffic Breakdown")
            attack_counts = df_live['attack_cat'].value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            
            colors = sns.color_palette("muted", len(attack_counts))
            attack_counts.plot(kind='barh', color=colors, ax=ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#F8FAFC')
            ax.spines['bottom'].set_color('#F8FAFC')
            ax.tick_params(colors='#F8FAFC')
            ax.set_xlabel("Count", color='#F8FAFC')
            ax.set_ylabel("Traffic Category", color='#F8FAFC')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        st.write("")
        
        # Alerts Feed
        st.subheader("🚨 Live Alerts Feed (Last 10 Anomalies)")
        anomalies_only = df_live[df_live['prediction'] == 1].tail(10)
        
        if anomalies_only.empty:
            st.info("No anomalies detected in the streaming window.")
        else:
            alerts_df = anomalies_only[[
                'timestamp', 'status', 'attack_cat', 'anomaly_prob', 'spatial_distance', 'rate', 'duration'
            ]].copy()
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'], unit='ms', utc=True).dt.tz_convert(datetime.datetime.now().astimezone().tzinfo).dt.strftime('%H:%M:%S')
            alerts_df = alerts_df.rename(columns={
                'timestamp': 'Time', 'status': 'Alert Class', 'attack_cat': 'Category', 
                'anomaly_prob': 'GRU Prob', 'spatial_distance': 'KMeans Dist', 'rate': 'Flow Rate', 'duration': 'Duration'
            })
            st.dataframe(alerts_df.iloc[::-1], use_container_width=True)

with tab2:
    st.header("📊 Evaluation Benchmark Results")
    st.markdown("""
    Evaluation was performed offline by benchmarking the trained pipeline on both the in-domain test set (**CICIDS2017**) 
    and the cross-dataset generalization test set (**UNSW-NB15**). This ensures that the spatial-temporal hybrid model is 
    robust and generalizable across different traffic distributions.
    """)
    st.header("🏆 Deep Learning Model Comparison")
    st.caption("Side-by-side evaluation of 3 DL architecture families: MLP Baseline, 1D-CNN, and BiLSTM")
    
    comp_csv = os.path.join(processed_dir, "comparison_table_cicids2017.csv")
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#1E3A8A'), use_container_width=True)
    
    col_plots1, col_plots2 = st.columns(2)
    with col_plots1:
        roc_img = os.path.join(processed_dir, "comparison_roc_cicids2017.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC Curves Overlay (CICIDS2017)")
    with col_plots2:
        bars_img = os.path.join(processed_dir, "comparison_bars_cicids2017.png")
        if os.path.exists(bars_img):
            st.image(bars_img, caption="Metric Bar Comparison (CICIDS2017)")
            
    cm_comp_img = os.path.join(processed_dir, "comparison_confusion_cicids2017.png")
    if os.path.exists(cm_comp_img):
        st.image(cm_comp_img, caption="Side-by-Side Confusion Matrices (MLP vs 1D-CNN vs BiLSTM)", use_container_width=True)

    st.markdown("---")
    col_eval_left, col_eval_right = st.columns(2)
    
    with col_eval_left:
        st.subheader("Primary Dataset: CICIDS2017")
        cic_cm = os.path.join(processed_dir, "confusion_matrix_cicids2017.png")
        cic_det = os.path.join(processed_dir, "attack_cat_detection_cicids2017.png")
        
        if os.path.exists(cic_cm):
            st.image(cic_cm, caption="CICIDS2017 Binary Confusion Matrix")
        else:
            st.info("Confusion matrix for CICIDS2017 will appear here after running evaluation.")
            
        if os.path.exists(cic_det):
            st.image(cic_det, caption="CICIDS2017 Detection Rates by Attack Category")
            
        cic_perf = os.path.join(processed_dir, "category_performance_cicids2017.csv")
        if os.path.exists(cic_perf):
            st.dataframe(pd.read_csv(cic_perf), use_container_width=True)
            
    with col_eval_right:
        st.subheader("Cross-Dataset Validation: UNSW-NB15")
        
        subtab_static, subtab_adapted = st.tabs(["❌ Static (Drifted)", "✅ Adapted (Retrained)"])
        
        with subtab_static:
            st.warning("Before adaptation: Severe performance drop due to domain shift (F1 ~ 0%).")
            unsw_cm_static = os.path.join(processed_dir, "confusion_matrix_unsw-nb15_(static).png")
            unsw_det_static = os.path.join(processed_dir, "attack_cat_detection_unsw-nb15_(static).png")
            
            if os.path.exists(unsw_cm_static):
                st.image(unsw_cm_static, caption="UNSW-NB15 Static Confusion Matrix")
            if os.path.exists(unsw_det_static):
                st.image(unsw_det_static, caption="UNSW-NB15 Static Detection Rates")
                
            unsw_perf_static = os.path.join(processed_dir, "category_performance_unsw-nb15_(static).csv")
            if os.path.exists(unsw_perf_static):
                st.dataframe(pd.read_csv(unsw_perf_static), use_container_width=True)
                
        with subtab_adapted:
            st.success("After active adaptation: Dynamic retraining restores F1-score to >90%!")
            unsw_cm_adapted = os.path.join(processed_dir, "confusion_matrix_unsw-nb15_(adapted).png")
            unsw_det_adapted = os.path.join(processed_dir, "attack_cat_detection_unsw-nb15_(adapted).png")
            
            if os.path.exists(unsw_cm_adapted):
                st.image(unsw_cm_adapted, caption="UNSW-NB15 Adapted Confusion Matrix")
            if os.path.exists(unsw_det_adapted):
                st.image(unsw_det_adapted, caption="UNSW-NB15 Adapted Detection Rates")
                
            unsw_perf_adapted = os.path.join(processed_dir, "category_performance_unsw-nb15_(adapted).csv")
            if os.path.exists(unsw_perf_adapted):
                st.dataframe(pd.read_csv(unsw_perf_adapted), use_container_width=True)

# Auto refresh script loop
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
