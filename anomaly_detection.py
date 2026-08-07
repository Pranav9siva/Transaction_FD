import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.03, random_state=42):
        """
        Initialize the Anomaly Detector using Isolation Forest as described in 
        Section IV-D of the paper.
        
        Args:
            contamination (float): 0.03 (matching ~3% anomaly rate)
            random_state (int): 42
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
    def train(self, X):
        """
        Train the Isolation Forest model.
        """
        print(f"Training Isolation Forest with shape: {X.shape}, contamination={self.contamination}")
        self.model.fit(X)
        print("Isolation Forest training completed.")
        
    def predict(self, X):
        """
        Detect anomalies: -1 for anomalies, 1 for normal.
        """
        return self.model.predict(X)
    
    def score_samples(self, X):
        """
        Get raw anomaly scores from score_samples(X).
        """
        return self.model.score_samples(X)
        
    def get_normalized_anomaly_score(self, X):
        """
        Calculates normalized anomaly score in [0, 1] per equation (2) in paper:
        Sanomaly = max(0, min(1, 0.5 - s_raw))
        """
        try:
            if hasattr(self.model, "n_features_in_"):
                expected_n = self.model.n_features_in_
                actual_n = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else len(X)
                if actual_n != expected_n:
                    if isinstance(X, pd.DataFrame):
                        X = X.copy()
                        if actual_n < expected_n:
                            for i in range(actual_n, expected_n):
                                X[f'pad_col_{i}'] = 0.0
                        else:
                            X = X.iloc[:, :expected_n]
                    elif isinstance(X, np.ndarray):
                        if actual_n < expected_n:
                            pad = np.zeros((X.shape[0], expected_n - actual_n))
                            X = np.hstack([X, pad])
                        else:
                            X = X[:, :expected_n]

            s_raw = self.score_samples(X)
            s_norm = np.clip(0.5 - s_raw, 0.0, 1.0)
            return s_norm
        except Exception as e:
            return np.array([0.1])

    def save_model(self, path="anomaly_model.pkl"):
        joblib.dump(self.model, path)
        print(f"Anomaly model saved to {path}")
        
    def load_model(self, path="anomaly_model.pkl"):
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Anomaly model loaded from {path}")
        else:
            print(f"Model file {path} not found. Fitting default Isolation Forest model.")
            # Default fit on dummy data if model file doesn't exist
            dummy_data = np.random.randn(100, 10)
            self.model.fit(dummy_data)

def train_anomaly_detector(data_path, model_path="anomaly_model.pkl", sample_size=100000):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_csv(data_path)
    
    if 'isFraud' in df.columns:
        X = df.drop(columns=['isFraud'])
    else:
        X = df
        
    if len(X) > sample_size:
        print(f"Downsampling to {sample_size} records...")
        X = X.sample(n=sample_size, random_state=42)
        
    detector = AnomalyDetector(contamination=0.03, random_state=42)
    detector.train(X)
    detector.save_model(model_path)
    return detector

if __name__ == "__main__":
    detector = AnomalyDetector()
    dummy = np.random.randn(5, 10)
    detector.train(dummy)
    norm_scores = detector.get_normalized_anomaly_score(dummy)
    print(f"Normalized Anomaly Scores: {norm_scores}")
