"""
ensemble_layer.py
Layer-1: Ensemble classifier (RandomForest, XGBoost, LightGBM).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

class EnsembleClassifier:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
        self.lgb = LGBMClassifier()
        self.meta = LogisticRegression()
        self.models = [self.rf, self.xgb, self.lgb]

    def fit(self, X, y):
        preds = []
        for model in self.models:
            model.fit(X, y)
            preds.append(model.predict_proba(X))
        stacked = np.hstack([p for p in preds])
        self.meta.fit(stacked, y)

    def predict(self, X):
        preds = [m.predict_proba(X) for m in self.models]
        stacked = np.hstack(preds)
        return self.meta.predict(stacked)

    def predict_proba(self, X):
        preds = [m.predict_proba(X) for m in self.models]
        stacked = np.hstack(preds)
        return self.meta.predict_proba(stacked)
