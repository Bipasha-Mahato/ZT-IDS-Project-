"""
anomaly_layer.py
Layer-2: DBSCAN anomaly detector for zero-day detection.
"""

import numpy as np
from sklearn.cluster import DBSCAN

class AnomalyDetector:
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X):
        """Fit DBSCAN and return cluster labels (-1 = anomaly)."""
        return self.model.fit_predict(X)

    def detect_outliers(self, X):
        """Return indices of detected anomalies."""
        labels = self.fit_predict(X)
        return np.where(labels == -1)[0]
