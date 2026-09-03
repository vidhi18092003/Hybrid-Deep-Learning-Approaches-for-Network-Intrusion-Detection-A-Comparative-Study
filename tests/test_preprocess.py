import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.preprocess import clean_data, normalize_cicids_attack_cat

class TestPreprocessing(unittest.TestCase):
    def test_clean_data_handles_nan_and_inf(self):
        # Create dummy df with NaN and Inf values
        data = {
            'duration': [100.0, np.inf, 300.0, np.nan, 500.0],
            'src_packets': [1.0, 2.0, np.nan, 4.0, 5.0],
            'label': [0, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        features = ['duration', 'src_packets']
        
        cleaned_df = clean_data(df, features)
        
        # Verify no NaN or Inf remains in feature columns
        for col in features:
            self.assertFalse(cleaned_df[col].isnull().any())
            self.assertFalse(np.isinf(cleaned_df[col]).any())
            
        # Verify the Inf value was replaced by median (which is 300.0)
        self.assertEqual(cleaned_df.loc[1, 'duration'], 300.0)
        # Verify the NaN value was replaced by median (which is 3.0)
        self.assertEqual(cleaned_df.loc[2, 'src_packets'], 3.0)

    def test_normalize_cicids_attack_cat(self):
        self.assertEqual(normalize_cicids_attack_cat("BENIGN"), "Benign")
        self.assertEqual(normalize_cicids_attack_cat("  benign  "), "Benign")
        self.assertEqual(normalize_cicids_attack_cat("DDoS-Friday"), "DDoS")
        self.assertEqual(normalize_cicids_attack_cat("DoS GoldenEye"), "DoS")
        self.assertEqual(normalize_cicids_attack_cat("PortScan"), "Portscan")
        self.assertEqual(normalize_cicids_attack_cat("Botnet-Friday"), "Botnet")
        self.assertEqual(normalize_cicids_attack_cat("Web Attack – Brute Force"), "WebAttack")
        self.assertEqual(normalize_cicids_attack_cat("FTP-Patator"), "Bruteforce")
        self.assertEqual(normalize_cicids_attack_cat("Infiltration"), "Infiltration")
        self.assertEqual(normalize_cicids_attack_cat(None), "Benign")
        self.assertEqual(normalize_cicids_attack_cat(123), "Benign")

if __name__ == '__main__':
    unittest.main()
