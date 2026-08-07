import csv
import json
import time
from datetime import datetime
import os

def stream_transactions(file_path, delay=1.0):
    """
    Generator that streams transactions from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        delay (float): Delay in seconds between each transaction.
        
    Yields:
        str: JSON string representing the transaction with a timestamp.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # Add timestamp to simulate real-time transaction
            row['timestamp'] = datetime.now().isoformat()
            
            # Convert to JSON object (string)
            json_record = json.dumps(row)
            
            yield json_record
            
            # Simulate streaming delay
            time.sleep(delay)

if __name__ == "__main__":
    # Default file path for demonstration
    file_path = "fraud_train_preprocessed.csv" if os.path.exists("fraud_train_preprocessed.csv") else r"c:\Users\sivap\Downloads\soft_eng\fraud_train_preprocessed.csv"
    
    print(f"Starting transaction stream from {file_path}...")
    try:
        for transaction in stream_transactions(file_path, delay=0.5):
            print(transaction)
    except KeyboardInterrupt:
        print("\nStream stopped.")
    except Exception as e:
        print(f"Error: {e}")
