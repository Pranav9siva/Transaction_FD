import json
import logging
import os
from datetime import datetime

class AlertSystem:
    def __init__(self, log_file="alerts.jsonl"):
        self.log_file = log_file
        self.file_logging_enabled = True

        try:
            with open(self.log_file, 'a') as f:
                pass
        except OSError as e:
            # Catch read-only filesystem errors in serverless environments like Vercel
            self.file_logging_enabled = False
            logging.warning("Read-only environment detected. File logging disabled.")

    def log_alert(self, transaction: dict, decision_result: dict):
        """
        Logs a flagged transaction to the alert file.
        """
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction.get('TransactionID', 'N/A'),
            'amount': transaction.get('TransactionAmt', 0.0),
            'decision': decision_result.get('decision', 'UNKNOWN'),
            'risk_score': decision_result.get('final_risk_score', 0.0),
            'top_factors': decision_result.get('explanation', {}).get('top_factors', {}),
            'raw_transaction': transaction
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(alert_record) + '\n')
            
        # Log high-level info to system log
        if decision_result.get('decision') == 'BLOCK':
            logging.critical(f"BLOCKED Transaction {alert_record['transaction_id']} - Score: {alert_record['risk_score']}")
            print(f"!!! ALERT: Transaction {alert_record['transaction_id']} BLOCKED (Score: {alert_record['risk_score']})")
        elif decision_result.get('decision') == 'REVIEW':
            logging.warning(f"REVIEW Transaction {alert_record['transaction_id']} - Score: {alert_record['risk_score']}")
            print(f"--- WARNING: Transaction {alert_record['transaction_id']} flagged for REVIEW")
