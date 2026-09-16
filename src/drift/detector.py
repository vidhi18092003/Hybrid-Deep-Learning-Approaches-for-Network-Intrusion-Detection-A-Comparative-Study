import numpy as np
import logging
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05, 
                 min_drift_features: int = 3, window_size: int = 1000):
        """
        concept drift detector based on Kolmogorov-Smirnov (KS) test.
        
        Args:
            reference_data: np.ndarray of shape [N, num_features] representing baseline normal traffic.
            alpha: float significance level for the KS test.
            min_drift_features: int minimum number of features that must drift to trigger retraining.
            window_size: int size of the rolling window of live data to monitor.
        """
        self.reference_data = reference_data
        self.alpha = alpha
        self.min_drift_features = min_drift_features
        self.window_size = window_size
        self.num_features = reference_data.shape[1]
        
        # Initialize rolling window
        self.live_window = []
        logger.info(f"Concept Drift Detector initialized. Reference shape: {reference_data.shape}, window_size: {window_size}")

    def update_reference(self, new_reference_data: np.ndarray):
        """Updates the reference dataset (e.g., after models have adapted)."""
        logger.info(f"Updating baseline reference data with shape {new_reference_data.shape}...")
        self.reference_data = new_reference_data

    def add_samples(self, X: np.ndarray) -> bool:
        """
        Adds new streaming samples to the rolling window and checks for drift.
        
        Returns:
            bool: True if drift is detected, False otherwise.
        """
        for sample in X:
            self.live_window.append(sample)
            # Maintain fixed rolling window size
            if len(self.live_window) > self.window_size:
                self.live_window.pop(0)
                
        # Only check for drift if we have a full window
        if len(self.live_window) >= self.window_size:
            return self.check_drift()
        return False

    def check_drift(self) -> bool:
        """
        Computes the KS-test between the live rolling window and the reference data.
        
        Returns:
            bool: True if concept drift is detected across min_drift_features features.
        """
        live_arr = np.array(self.live_window)
        drift_count = 0
        p_values = []
        
        for i in range(self.num_features):
            ref_col = self.reference_data[:, i]
            live_col = live_arr[:, i]
            
            # Run two-sample Kolmogorov-Smirnov test
            stat, p_val = ks_2samp(ref_col, live_col)
            p_values.append(p_val)
            
            if p_val < self.alpha:
                drift_count += 1
                
        # Log drift diagnostics
        if drift_count >= self.min_drift_features:
            logger.warning(
                f"CONCEPT DRIFT DETECTED: {drift_count}/{self.num_features} features drifted "
                f"(min required: {self.min_drift_features}). p-values: {[f'{p:.4f}' for p in p_values]}"
            )
            return True
        else:
            logger.debug(
                f"Drift check: {drift_count}/{self.num_features} features drifted. "
                f"p-values: {[f'{p:.4f}' for p in p_values]}"
            )
            return False

    def clear_live_window(self):
        """Clears the live window, typically called after a retraining event."""
        logger.info("Clearing live rolling window post-retraining.")
        self.live_window = []
