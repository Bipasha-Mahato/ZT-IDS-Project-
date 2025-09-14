"""
main.py
Main driver script for ZT-IDS.
"""

from preprocessing import prepare_data
from ensemble_layer import EnsembleClassifier
from anomaly_layer import AnomalyDetector
from visualization import plot_confusion_matrix

def run_pipeline(dataset_path):
    # Step 1: Data prep
    X_train, X_test, y_train, y_test = prepare_data(dataset_path, label_col="Label")

    # Step 2: Layer-1 Ensemble
    clf = EnsembleClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Step 3: Evaluate Layer-1
    plot_confusion_matrix(y_test, y_pred)

    # Step 4: Layer-2 Anomaly detection (applied only to benign flows from Layer-1)
    detector = AnomalyDetector(eps=0.7, min_samples=10)
    outliers = detector.detect_outliers(X_test)

    print(f"Detected {len(outliers)} anomalies (Layer-2 Zero-Day).")

if __name__ == "__main__":
    dataset_path = "data/cicids_sample.csv"  # <-- replace with actual
    run_pipeline(dataset_path)
