"""
preprocessing.py
Feature engineering and dataset preparation for ZT-IDS.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset(path):
    """Load dataset from CSV or other formats."""
    df = pd.read_csv(path)
    return df

def feature_reduction(df, drop_cols=None):
    """Apply basic feature reduction (variance threshold / correlation)."""
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def scale_features(X):
    """Standardize features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def prepare_data(path, label_col="Label", test_size=0.2, drop_cols=None):
    """Full pipeline: load, preprocess, split."""
    df = load_dataset(path)
    df = feature_reduction(df, drop_cols)
    
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    X_scaled = scale_features(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
