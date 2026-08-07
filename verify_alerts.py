from alert_system import AlertSystem
import time
import random

def verify_alerts():
    print("Initializing Alert System...")
    alert_system = AlertSystem()
    
    # Mock Data
    transactions = [
        {'TransactionID': 'TXN_1001', 'TransactionAmt': 5000.0, 'card1': 1234},
        {'TransactionID': 'TXN_1002', 'TransactionAmt': 25.0, 'card1': 5678},
        {'TransactionID': 'TXN_1003', 'TransactionAmt': 9999.0, 'card1': 9999},
    ]
    
    decisions = [
        {'decision': 'BLOCK', 'final_risk_score': 0.95, 'explanation': {'top_factors': {'TransactionAmt': 0.8, 'V200': 0.1}}},
        {'decision': 'LEGIT', 'final_risk_score': 0.10, 'explanation': {'top_factors': {'TransactionAmt': -0.2}}},
        {'decision': 'REVIEW', 'final_risk_score': 0.65, 'explanation': {'top_factors': {'C1': 0.4, 'D1': 0.2}}}
    ]
    
    print("Generating mock alerts...")
    for i in range(3):
        alert_system.log_alert(transactions[i], decisions[i])
        time.sleep(0.1)
        
    print("Done. Checking alerts.jsonl...")
    with open('alerts.jsonl', 'r') as f:
        lines = f.readlines()
        print(f"Found {len(lines)} alerts in log file.")
        for line in lines:
            print(line.strip())

if __name__ == "__main__":
    verify_alerts()
