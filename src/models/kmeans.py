import pickle
import logging
import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

class SpatialAnomalyFilter:
    def __init__(self, n_clusters: int = 8, percentile_threshold: float = 99.0):
        self.n_clusters = n_clusters
        self.percentile_threshold = percentile_threshold
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        self.threshold = 0.0
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """Train K-Means on normal/benign data to establish baseline centroids."""
        logger.info(f"Fitting MiniBatchKMeans with {self.n_clusters} clusters on {X.shape[0]} normal flows...")
        self.kmeans.fit(X)
        self.is_fitted = True
        
        # Calculate distances to nearest centroid for normal data to establish threshold
        distances = self.compute_distances(X)
        self.threshold = np.percentile(distances, self.percentile_threshold)
        logger.info(f"K-Means fitted. Dynamic distance threshold ({self.percentile_threshold}th percentile): {self.threshold:.4f}")

    def partial_fit(self, X: np.ndarray):
        """Incrementally update cluster centroids with new streaming data."""
        if not self.is_fitted:
            logger.warning("K-Means is not fitted yet. Calling fit instead of partial_fit.")
            self.fit(X)
            return
        
        logger.info(f"Incrementally updating K-Means centroids with {X.shape[0]} flows...")
        self.kmeans.partial_fit(X)
        
        # Re-evaluate threshold on the new data
        distances = self.compute_distances(X)
        # We blend the threshold slightly (e.g., exponential moving average) to prevent threshold shift
        new_thresh = np.percentile(distances, self.percentile_threshold)
        self.threshold = 0.9 * self.threshold + 0.1 * new_thresh
        logger.info(f"Updated K-Means centroids. Blended distance threshold: {self.threshold:.4f}")

    def compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate the Euclidean distance of each sample to its nearest centroid."""
        if not self.is_fitted:
            raise ValueError("K-Means model is not fitted yet.")
        
        # transform returns distance to all centroids
        distances_to_centroids = self.kmeans.transform(X)
        # Return the minimum distance (nearest centroid) for each flow
        return np.min(distances_to_centroids, axis=1)

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Flag spatial outliers where distance to nearest centroid exceeds the threshold.
        
        Returns a boolean array where True indicates a candidate anomaly.
        """
        distances = self.compute_distances(X)
        return distances > self.threshold

    def save(self, filepath: str):
        """Saves the K-Means filter state to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'threshold': self.threshold,
                'n_clusters': self.n_clusters,
                'percentile_threshold': self.percentile_threshold,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"SpatialAnomalyFilter saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SpatialAnomalyFilter':
        """Loads the K-Means filter state from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(n_clusters=data['n_clusters'], percentile_threshold=data['percentile_threshold'])
        obj.kmeans = data['kmeans']
        obj.threshold = data['threshold']
        obj.is_fitted = data['is_fitted']
        logger.info(f"SpatialAnomalyFilter loaded from {filepath}. Threshold: {obj.threshold:.4f}")
        return obj
