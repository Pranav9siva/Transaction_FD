import pandas as pd
import numpy as np
import os
import random

class SyntheticTransactionGenerator:
    """
    Template-based perturbation synthetic transaction generator as described in 
    'Real-Time Transaction Fraud Detection System Using Machine Learning and Stream Processing'.
    
    1. Select a random real row from the corresponding class in the test set as a template.
    2. Perturb 5-10 of the 110 continuous features with Gaussian noise:
       x_i' = clip(x_i + N(0, sigma_i * s), x_min, x_max) where s ~ U(0.08, 0.12)
    3. Each binary feature has a 2% independent flip probability.
    """
    def __init__(self, data_path="fraud_test_preprocessed.csv"):
        self.data_path = data_path
        self.templates_legit = []
        self.templates_fraud = []
        self.continuous_cols = []
        self.binary_cols = []
        self.feature_cols = []
        self.stats = {}
        
        self._load_and_prepare_templates()

    def _load_and_prepare_templates(self):
        # Fallback path check
        path = self.data_path
        if not os.path.exists(path):
            path = "fraud_train_preprocessed.csv"
        
        if os.path.exists(path):
            print(f"[SyntheticGenerator] Loading template dataset from {path}...")
            # Load sample for performance
            df = pd.read_csv(path, nrows=5000)
            
            if 'isFraud' in df.columns:
                legit_df = df[df['isFraud'] == 0]
                fraud_df = df[df['isFraud'] == 1]
                
                features_df = df.drop(columns=['isFraud'])
            else:
                legit_df = df
                fraud_df = df
                features_df = df
                
            self.feature_cols = list(features_df.columns)
            
            # Identify continuous vs binary columns
            for col in self.feature_cols:
                unique_vals = features_df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    self.binary_cols.append(col)
                else:
                    self.continuous_cols.append(col)
                    
            # Compute stats (std, min, max) for continuous columns
            for col in self.continuous_cols:
                col_data = features_df[col].dropna()
                if len(col_data) > 0:
                    self.stats[col] = {
                        'std': float(col_data.std()) if col_data.std() > 0 else 1.0,
                        'min': float(col_data.min()),
                        'max': float(col_data.max())
                    }
                else:
                    self.stats[col] = {'std': 1.0, 'min': 0.0, 'max': 1000.0}
                    
            self.templates_legit = legit_df.to_dict('records')
            self.templates_fraud = fraud_df.to_dict('records')
            print(f"[SyntheticGenerator] Loaded {len(self.templates_legit)} legit and {len(self.templates_fraud)} fraud templates.")
        else:
            print("[SyntheticGenerator] No dataset found; using mock schema generator.")
            self._create_mock_schema()

    def _create_mock_schema(self):
        # Default fallback schema with 110 continuous and 159 binary features (total 269)
        self.continuous_cols = [f'C_{i}' for i in range(1, 111)] + ['TransactionAmt']
        self.binary_cols = [f'B_{i}' for i in range(1, 159)]
        self.feature_cols = self.continuous_cols + self.binary_cols
        
        mock_record_legit = {col: 0.0 for col in self.feature_cols}
        mock_record_legit['TransactionAmt'] = 50.0
        mock_record_legit['card1'] = 1234
        mock_record_legit['card2'] = 5678
        
        mock_record_fraud = {col: 1.0 for col in self.feature_cols}
        mock_record_fraud['TransactionAmt'] = 2500.0
        mock_record_fraud['card1'] = 9999
        mock_record_fraud['card2'] = 8888
        
        self.templates_legit = [mock_record_legit]
        self.templates_fraud = [mock_record_fraud]
        
        for col in self.continuous_cols:
            self.stats[col] = {'std': 10.0, 'min': 0.0, 'max': 10000.0}

    def generate_transaction(self, force_fraud=None):
        """
        Generates a synthetic transaction using template-based perturbation.
        
        Target ratio: ~28% fraud in streaming mode to match Table IV in paper.
        """
        if force_fraud is True:
            template = random.choice(self.templates_fraud or self.templates_legit).copy()
        elif force_fraud is False:
            template = random.choice(self.templates_legit).copy()
        else:
            # Weighted choice to yield ~28% fraud output distribution after perturbation
            is_fraud = random.random() < 0.28
            pool = self.templates_fraud if (is_fraud and self.templates_fraud) else self.templates_legit
            template = random.choice(pool).copy()

        # Remove target if present
        template.pop('isFraud', None)

        # 1. Perturb 5-10 continuous features
        if self.continuous_cols:
            num_to_perturb = min(random.randint(5, 10), len(self.continuous_cols))
            selected_cont = random.sample(self.continuous_cols, num_to_perturb)
            
            s = random.uniform(0.08, 0.12) # Noise scale s ~ U(0.08, 0.12)
            
            for col in selected_cont:
                val = float(template.get(col, 0.0) or 0.0)
                st = self.stats.get(col, {'std': 1.0, 'min': 0.0, 'max': 10000.0})
                noise = np.random.normal(0, st['std'] * s)
                perturbed_val = np.clip(val + noise, st['min'], st['max'])
                template[col] = float(perturbed_val)

        # 2. Binary features: 2% independent flip probability
        for col in self.binary_cols:
            if random.random() < 0.02:
                val = template.get(col, 0)
                template[col] = 1 - int(val or 0)

        # Ensure essential identifiers and timestamps exist
        if 'TransactionID' not in template or not template['TransactionID']:
            template['TransactionID'] = f"SYN-{random.randint(100000, 999999)}"
        if 'card1' not in template or pd.isna(template['card1']):
            template['card1'] = random.randint(1000, 9999)
        if 'card2' not in template or pd.isna(template['card2']):
            template['card2'] = random.randint(1000, 9999)
        if 'TransactionAmt' not in template or pd.isna(template['TransactionAmt']):
            template['TransactionAmt'] = round(random.uniform(10.0, 1500.0), 2)
            
        return template

if __name__ == "__main__":
    gen = SyntheticTransactionGenerator()
    tx = gen.generate_transaction()
    print(f"Generated synthetic transaction ID: {tx['TransactionID']}, Amt: {tx['TransactionAmt']}")
