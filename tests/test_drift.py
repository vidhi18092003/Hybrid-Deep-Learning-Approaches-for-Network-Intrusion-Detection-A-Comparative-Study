import unittest
import numpy as np
from src.drift.detector import DriftDetector

class TestDriftDetector(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create baseline reference data (10 features, 1000 samples, normal distribution)
        self.ref_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10))

    def test_no_drift_when_distributions_match(self):
        detector = DriftDetector(
            reference_data=self.ref_data,
            alpha=0.05,
            min_drift_features=3,
            window_size=200
        )
        
        # Generate new samples from the exact same distribution
        new_samples = np.random.normal(loc=0.0, scale=1.0, size=(200, 10))
        
        # Add samples and check for drift
        drift_detected = detector.add_samples(new_samples)
        
        # With the same distribution, drift should not be triggered
        self.assertFalse(drift_detected)

    def test_drift_detected_when_distribution_shifts(self):
        detector = DriftDetector(
            reference_data=self.ref_data,
            alpha=0.01, # Strict significance level
            min_drift_features=3,
            window_size=200
        )
        
        # Shift 5 features significantly (mean shifts from 0 to 2)
        shifted_samples = np.random.normal(loc=0.0, scale=1.0, size=(200, 10))
        for col_idx in range(5):
            shifted_samples[:, col_idx] += 2.0
            
        drift_detected = detector.add_samples(shifted_samples)
        
        # Since 5 features shifted (>= 3 minimum), drift should be detected
        self.assertTrue(drift_detected)

if __name__ == '__main__':
    unittest.main()
