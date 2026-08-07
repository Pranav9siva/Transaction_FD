import pandas as pd
import numpy as np
from collections import defaultdict, deque
from datetime import datetime

class TransactionPreprocessor:
    def __init__(self):
        """
        Initialize the preprocessor with history tracking for feature engineering.
        """
        # User history: user_id -> list of (timestamp, amount, location)
        self.user_history = defaultdict(lambda: deque(maxlen=20))
        
        # Global stats for simple normalization (in a real system, these would be loaded from a store)
        self.amount_stats = {'n': 0, 'mean': 0.0, 'm2': 0.0}
        
        # Categorical encoders (simplified for demonstration)
        self.label_encoders = defaultdict(dict)
        
    def fit(self, records):
        """
        Fit scalers and encoders on a batch of records.
        """
        # In a streaming context, we might skip this or use partial_fit
        pass

    def process_record(self, record):
        """
        Process a single transaction record.
        
        1. Clean missing values.
        2. Encode categorical features.
        3. Generate fraud features (freq, deviation, loc change).
        4. Normalize numeric values.
        
        Args:
            record (dict): Raw transaction dictionary.
            
        Returns:
            dict: Processed feature vector.
        """
        # 1. Clean missing values
        record = self._clean_missing(record)
        
        # Extract Key Fields for Feature Engineering
        # Assuming keys exist or using defaults
        user_id = record.get('card1', record.get('user_id', 'unknown'))
        try:
            amount = float(record.get('TransactionAmt', record.get('amount', 0.0)))
        except ValueError:
            amount = 0.0
            
        location = record.get('addr1', record.get('location', 0))
        timestamp_str = record.get('timestamp', datetime.now().isoformat())
        
        # 2. Encode Categorical Features
        # Example: One-hot encode 'ProductCD' if present, else just label encode user_id
        encoded_user = self._encode_categorical('user_id', user_id)
        
        # 3. Generate Fraud-Related Features
        fraud_features = self._generate_fraud_features(user_id, amount, location, timestamp_str)
        
        # 4. Normalize Numeric Values (Amount)
        normalized_amount = self._update_and_scale_amount(amount)
        
        # Assemble Feature Vector
        feature_vector = {
            'encoded_user_id': encoded_user,
            'normalized_amount': normalized_amount,
            'transaction_freq_24h': fraud_features['freq'],
            'amount_deviation_score': fraud_features['dev'],
            'location_change_flag': fraud_features['loc_change'],
            # Pass through original timestamp for reference
            'timestamp': timestamp_str
        }
        
        # Merge with other numeric columns from record if needed
        # feature_vector.update({k: v for k, v in record.items() if isinstance(v, (int, float))})
        
        return feature_vector

    def _clean_missing(self, record):
        cleaned = {}
        for k, v in record.items():
            if v is None or v == '' or str(v).lower() == 'nan':
                cleaned[k] = 0  # Impute with 0 for simplicity
            else:
                cleaned[k] = v
        return cleaned

    def _encode_categorical(self, feature_name, value):
        # Simple dynamic label encoding
        if value not in self.label_encoders[feature_name]:
            self.label_encoders[feature_name][value] = len(self.label_encoders[feature_name])
        return self.label_encoders[feature_name][value]

    def _generate_fraud_features(self, user_id, amount, location, timestamp_str):
        history = self.user_history[user_id]
        
        try:
            current_ts = pd.to_datetime(timestamp_str)
        except:
            current_ts = datetime.now()

        # Frequency: Count transactions in last 24 hours (simulated window)
        # Here we just count history length for simplicity in this stream
        freq = len(history)
        
        # Amount Deviation: Z-score against user's history
        if history:
            amounts = [h[1] for h in history]
            mean_amt = np.mean(amounts)
            std_amt = np.std(amounts) + 1e-6 # Avoid div by zero
            dev = (amount - mean_amt) / std_amt
        else:
            dev = 0.0
            
        # Location Change
        loc_change = 0
        if history:
            last_loc = history[-1][2]
            if location != last_loc:
                loc_change = 1
                
        # Update history
        history.append((current_ts, amount, location))
        
        return {
            'freq': freq,
            'dev': dev,
            'loc_change': loc_change
        }

    def _update_and_scale_amount(self, amount):
        # Online Welford's algorithm for global scaling
        self.amount_stats['n'] += 1
        delta = amount - self.amount_stats['mean']
        self.amount_stats['mean'] += delta / self.amount_stats['n']
        delta2 = amount - self.amount_stats['mean']
        self.amount_stats['m2'] += delta * delta2
        
        if self.amount_stats['n'] < 2:
            return 0.0
            
        variance = self.amount_stats['m2'] / (self.amount_stats['n'] - 1)
        std_dev = np.sqrt(variance) + 1e-6
        
        return (amount - self.amount_stats['mean']) / std_dev
