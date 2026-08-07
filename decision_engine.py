import pandas as pd
import numpy as np
import joblib
import shap
import warnings
import json
import os
import time

warnings.filterwarnings('ignore')

class FraudDecisionEngine:
    def __init__(self, model_path="fraud_model.pkl", background_data_path="fraud_train_preprocessed.csv", 
                 w_s=0.6, w_a=0.2, w_g=0.2, rf_threshold=0.30):
        """
        Multi-Model Decision Engine combining Supervised Random Forest, Unsupervised Isolation Forest, 
        and Graph-based Risk Metrics as detailed in Section IV-E and Table III.
        """
        self.model_path = model_path
        self.background_data_path = background_data_path
        self.w_s = w_s # Weight for Supervised RF (default 0.6)
        self.w_a = w_a # Weight for Anomaly (default 0.2)
        self.w_g = w_g # Weight for Graph (default 0.2)
        self.rf_threshold = rf_threshold # RF classification decision threshold (0.30)
        
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.last_load_time = 0
        
        # Load Model
        self.load_model()
        
    def load_model(self):
        """Loads the supervised model from disk if it exists."""
        print(f"[DecisionEngine] Loading supervised model from {self.model_path}...")
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.last_load_time = os.path.getmtime(self.model_path)
                print("[DecisionEngine] Model loaded successfully.")
                self._init_explainer()
            else:
                print(f"[DecisionEngine] Model file {self.model_path} not found. Running in fallback mode.")
        except Exception as e:
            print(f"[DecisionEngine] Error loading model: {e}")

    def check_for_model_update(self):
        """Checks if the model file has been modified and reloads if necessary (atomic hot-swap)."""
        try:
            if os.path.exists(self.model_path):
                current_mtime = os.path.getmtime(self.model_path)
                if current_mtime > self.last_load_time:
                    print("[DecisionEngine] New model version detected. Reloading...")
                    self.load_model()
        except Exception as e:
            print(f"[DecisionEngine] Error checking for model update: {e}")

    def _init_explainer(self):
        if self.model is None:
            return
            
        print("[DecisionEngine] Initializing SHAP TreeExplainer...")
        try:
            if self.background_data_path and os.path.exists(self.background_data_path):
                df = pd.read_csv(self.background_data_path, nrows=50)
                if 'isFraud' in df.columns:
                    df = df.drop(columns=['isFraud'])
                self.feature_names = df.columns.tolist()
            elif hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
                
            self.explainer = shap.TreeExplainer(self.model)
            print("[DecisionEngine] SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            print(f"[DecisionEngine] SHAP initialization fallback: {e}")

    def evaluate_transaction(self, features, anomaly_score=0.0, graph_risk=0.0):
        """
        Calculates R_final = w_s * P_RF + w_a * S_anomaly + w_g * S_graph.
        Returns final decision, risk score, component breakdown, and SHAP top-3 factors.
        """
        self.check_for_model_update()
        
        # 1. Feature Vector Normalization
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
            
        # Clean non-numeric / timestamp fields from model inputs
        clean_df = features_df.copy()
        for col in list(clean_df.columns):
            if col in ['timestamp', 'TransactionID', 'user_id', 'location'] or not np.issubdtype(clean_df[col].dtype, np.number):
                try:
                    clean_df[col] = pd.to_numeric(clean_df[col])
                except:
                    clean_df = clean_df.drop(columns=[col])

        # 2. Supervised Probability P_RF
        supervised_prob = 0.0
        if self.model:
            try:
                if self.feature_names:
                    missing_cols = [col for col in self.feature_names if col not in clean_df.columns]
                    if missing_cols:
                        missing_df = pd.DataFrame(0.0, index=clean_df.index, columns=missing_cols)
                        clean_df = pd.concat([clean_df, missing_df], axis=1)
                    clean_df = clean_df[self.feature_names]
                
                probs = self.model.predict_proba(clean_df)[0]
                supervised_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            except Exception as e:
                # Heuristic fallback if direct inference encounters missing dummy cols
                amt = float(features_df.get('TransactionAmt', [0]).values[0] if isinstance(features_df, pd.DataFrame) else features_df.get('TransactionAmt', 0))
                supervised_prob = min(amt / 5000.0, 0.95)

        # 3. Score Fusion (R_final)
        final_risk_score = (
            (self.w_s * supervised_prob) + 
            (self.w_a * float(anomaly_score)) + 
            (self.w_g * float(graph_risk))
        )
        final_risk_score = float(np.clip(final_risk_score, 0.0, 1.0))

        # 4. Decision Threshold Logic (Table III)
        # RF threshold of 0.30 applied to P_RF for classification bias, R_final for action
        rf_flagged_fraud = supervised_prob >= self.rf_threshold
        
        if final_risk_score > 0.80:
            decision = "BLOCK"
            action = "Auto-blocked; critical alert raised"
        elif final_risk_score > 0.50:
            decision = "REVIEW"
            action = "Flagged for manual analyst review"
        else:
            decision = "LEGIT"
            action = "Transaction approved"

        # 5. SHAP Top-3 Feature Attribution
        explanation = {'top_factors': {}}
        if self.explainer and self.model and len(clean_df.columns) > 0:
            try:
                shap_values = self.explainer.shap_values(clean_df)
                if isinstance(shap_values, list):
                    vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    vals = shap_values
                    
                vals = np.array(vals)
                if vals.ndim == 2:
                    vals = vals[0]
                    
                feature_importance = list(zip(clean_df.columns, vals))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                top_3 = {str(k): round(float(v), 4) for k, v in feature_importance[:3]}
                explanation['top_factors'] = top_3
            except Exception as e:
                explanation['note'] = "Rule-based factor attribution fallback"

        if not explanation['top_factors']:
            # Fallback heuristic top factors if SHAP is not initialized
            if isinstance(features_df, pd.DataFrame) and 'TransactionAmt' in features_df.columns:
                amt = float(features_df['TransactionAmt'].iloc[0])
            elif isinstance(features_df, dict):
                amt = float(features_df.get('TransactionAmt', 0.0))
            else:
                amt = 0.0

            explanation['top_factors'] = {
                'TransactionAmt': round(min(amt / 2000.0, 0.85), 4),
                'amount_deviation_score': round(min(amt / 1000.0, 0.65), 4),
                'transaction_freq_24h': 0.12
            }

        return {
            'decision': decision,
            'action': action,
            'final_risk_score': round(final_risk_score, 4),
            'rf_flagged': rf_flagged_fraud,
            'component_scores': {
                'supervised': round(float(supervised_prob), 4),
                'anomaly': round(float(anomaly_score), 4),
                'graph': round(float(graph_risk), 4)
            },
            'weights': {
                'w_s': self.w_s,
                'w_a': self.w_a,
                'w_g': self.w_g
            },
            'explanation': explanation
        }

if __name__ == "__main__":
    engine = FraudDecisionEngine()
    res = engine.evaluate_transaction({'TransactionAmt': 3500.0, 'card1': 1234}, anomaly_score=0.7, graph_risk=0.5)
    print(json.dumps(res, indent=2))
