import pandas as pd
import joblib
import os
import shutil
from datetime import datetime
from model_training import train_model

class FeedbackLoop:
    def __init__(self, feedback_file="feedback_data.csv", training_data="fraud_train_preprocessed.csv", model_path="fraud_model.pkl"):
        self.feedback_file = feedback_file
        self.training_data = training_data
        self.model_path = model_path
        
        # Initialize feedback file if not exists
        if not os.path.exists(self.feedback_file):
            pd.DataFrame(columns=['TransactionID', 'isFraud']).to_csv(self.feedback_file, index=False)

    def log_feedback(self, transaction_id, is_fraud, features=None):
        """
        Stores confirmed fraud outcomes.
        If features are provided, they can be saved for retraining.
        For simplicity, we assume we can link back to original data or features are passed.
        """
        # Create record
        record = {
            'TransactionID': transaction_id,
            'isFraud': 1 if is_fraud else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # If features provided, flatten and include
        if features:
            record.update(features)
            
        # Append to feedback file
        df = pd.DataFrame([record])
        
        # If file exists, append without header
        if os.path.exists(self.feedback_file) and os.path.getsize(self.feedback_file) > 0:
            df.to_csv(self.feedback_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.feedback_file, mode='w', header=True, index=False)
            
        print(f"Feedback logged for Transaction {transaction_id}: {'Fraud' if is_fraud else 'Legit'}")

    def retrain(self):
        """
        Periodically retrains models using new data.
        Updates models without stopping the system (atomic swap).
        """
        print("Starting model retraining...")
        
        # 1. Load Original Data
        if not os.path.exists(self.training_data):
            print(f"Error: Training data {self.training_data} not found.")
            return
            
        df_train = pd.read_csv(self.training_data)
        
        # 2. Load Feedback Data
        if os.path.exists(self.feedback_file) and os.path.getsize(self.feedback_file) > 0:
            try:
                # Read feedback with all columns
                df_feedback = pd.read_csv(self.feedback_file)
                
                # Filter for columns that match training data
                common_cols = [c for c in df_train.columns if c in df_feedback.columns]
                
                if 'isFraud' in common_cols:
                    # Append feedback to training data
                    df_combined = pd.concat([df_train, df_feedback[common_cols]], ignore_index=True)
                    print(f"Combined data: {len(df_train)} original + {len(df_feedback)} feedback = {len(df_combined)} total.")
                else:
                    print("Warning: Feedback data missing target or columns. Using original data only.")
                    df_combined = df_train
            except Exception as e:
                print(f"Error reading feedback file: {e}")
                df_combined = df_train
        else:
            print("No feedback data found. Retraining on original data.")
            df_combined = df_train

        # 3. Retrain Model
        # Save to a temporary path first to avoid locking issues/race conditions
        temp_model_path = self.model_path + ".tmp"
        
        try:
            # We use the train_model function but need to pass the dataframe or save it to a temp csv
            # Since train_model expects a path, we'll save our combined df to a temp csv
            temp_csv_path = "temp_training_data.csv"
            df_combined.to_csv(temp_csv_path, index=False)
            
            # Train
            train_model(temp_csv_path, model_output_path=temp_model_path)
            
            # Cleanup temp csv
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                
            # 4. Atomic Update (Hot Swap)
            # Rename temp model to actual model path
            # On Windows, os.rename might fail if destination exists, so we use shutil.move or os.replace
            if os.path.exists(self.model_path):
                os.replace(temp_model_path, self.model_path)
            else:
                os.rename(temp_model_path, self.model_path)
                
            print(f"Successfully updated model at {self.model_path}")
            
        except Exception as e:
            print(f"Retraining failed: {e}")
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)

if __name__ == "__main__":
    # Test Feedback Loop
    loop = FeedbackLoop()
    
    # Mock Feedback
    print("Logging mock feedback...")
    loop.log_feedback("TXN_MOCK_1", is_fraud=True, features={'TransactionAmt': 5000, 'card1': 1001})
    loop.log_feedback("TXN_MOCK_2", is_fraud=False, features={'TransactionAmt': 50, 'card1': 1002})
    
    # Trigger Retraining
    loop.retrain()
