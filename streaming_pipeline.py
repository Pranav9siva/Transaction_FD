import os
import json
import time
import threading
import queue
import pandas as pd
import numpy as np
from datetime import datetime

from synthetic_generator import SyntheticTransactionGenerator
from preprocessing import TransactionPreprocessor
from decision_engine import FraudDecisionEngine
from anomaly_detection import AnomalyDetector
from alert_system import AlertSystem
from graph_fraud import FraudGraphDetector

try:
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False
    print("[StreamingPipeline] kafka-python not found. Falling back to in-memory queue.")

class FraudStreamingPipeline:
    def __init__(self, bootstrap_servers='localhost:9092', topic='financial_transactions'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.running = False
        self.producer_thread = None
        self.mock_queue = queue.Queue(maxsize=1000)
        self.sse_listeners = []
        self.stats = {
            'total_processed': 0,
            'total_blocked': 0,
            'total_review': 0,
            'total_legit': 0,
            'recent_latency_ms': 0.0,
            'start_time': datetime.now().isoformat()
        }

        print("[StreamingPipeline] Initializing Pipeline Components...")
        self.generator = SyntheticTransactionGenerator()
        self.preprocessor = TransactionPreprocessor()
        self.decision_engine = FraudDecisionEngine()
        self.anomaly_detector = AnomalyDetector(contamination=0.03, random_state=42)
        
        # Fit anomaly detector on initial dummy sample if needed
        dummy_sample = pd.DataFrame(np.random.randn(100, 10))
        self.anomaly_detector.train(dummy_sample)
        
        self.alert_system = AlertSystem()
        self.graph_detector = FraudGraphDetector(sender_col='card1', receiver_col='card2')

    def add_sse_listener(self, listener_queue):
        self.sse_listeners.append(listener_queue)

    def remove_sse_listener(self, listener_queue):
        if listener_queue in self.sse_listeners:
            self.sse_listeners.remove(listener_queue)

    def broadcast_sse(self, payload):
        for q in list(self.sse_listeners):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def start_pipeline(self, delay=1.5):
        if self.running:
            print("[StreamingPipeline] Pipeline is already running.")
            return
            
        self.running = True
        self.producer_thread = threading.Thread(target=self._producer_loop, args=(delay,))
        self.producer_thread.daemon = True
        self.producer_thread.start()

        consumer_thread = threading.Thread(target=self._consumer_loop)
        consumer_thread.daemon = True
        consumer_thread.start()
        print("[StreamingPipeline] Pipeline daemon threads started.")

    def stop_pipeline(self):
        self.running = False
        print("[StreamingPipeline] Pipeline stopped.")

    def _producer_loop(self, delay):
        producer = None
        if HAS_KAFKA:
            try:
                producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
            except Exception as e:
                print(f"[StreamingPipeline] Kafka connect failed ({e}); using mock queue.")

        while self.running:
            raw_record = self.generator.generate_transaction()
            if producer:
                try:
                    producer.send(self.topic, raw_record)
                except Exception:
                    self.mock_queue.put(raw_record)
            else:
                try:
                    self.mock_queue.put(raw_record, timeout=0.5)
                except queue.Full:
                    pass
                    
            time.sleep(delay)

    def _consumer_loop(self):
        while self.running:
            try:
                raw_record = self.mock_queue.get(timeout=1.0)
                self.process_transaction(raw_record)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[StreamingPipeline] Consumer error: {e}")

    def process_transaction(self, raw_record, source="STREAM"):
        """
        Core Processing Engine:
        Latency target < 200 ms. Generates 28-field JSON event payload for SSE and API.
        """
        start_t = time.time()

        # 1. Preprocessing
        features = self.preprocessor.process_record(raw_record)

        # 2. Anomaly Detection
        feat_df = pd.DataFrame([features])
        # Clean numeric
        num_df = feat_df.select_dtypes(include=[np.number])
        if num_df.empty:
            num_df = pd.DataFrame({'TransactionAmt': [float(raw_record.get('TransactionAmt', 0))]})
            
        norm_anomaly_score = float(self.anomaly_detector.get_normalized_anomaly_score(num_df)[0])

        # 3. Graph Analysis
        sender = raw_record.get('card1', 'UNKNOWN')
        receiver = raw_record.get('card2', 'UNKNOWN')
        amt = float(raw_record.get('TransactionAmt', 0.0))
        tx_id = str(raw_record.get('TransactionID', 'N/A'))
        
        self.graph_detector.add_transaction(sender, receiver, amount=amt, transaction_id=tx_id)
        graph_risk = self.graph_detector.get_node_risk(sender)
        
        hub_nodes = self.graph_detector.detect_highly_connected_nodes(threshold=10)
        cycles = self.graph_detector.get_cycles()
        
        hub_flag = sender in hub_nodes or receiver in hub_nodes
        cycle_flag = any(sender in c for c in cycles)

        # 4. Multi-Model Decision Engine
        decision_res = self.decision_engine.evaluate_transaction(
            features, 
            anomaly_score=norm_anomaly_score, 
            graph_risk=graph_risk
        )

        # 5. Alerting & Stats
        if decision_res['decision'] in ['BLOCK', 'REVIEW']:
            self.alert_system.log_alert(raw_record, decision_res)

        elapsed_ms = round((time.time() - start_t) * 1000.0, 2)
        
        # Update metrics
        self.stats['total_processed'] += 1
        if decision_res['decision'] == 'BLOCK':
            self.stats['total_blocked'] += 1
        elif decision_res['decision'] == 'REVIEW':
            self.stats['total_review'] += 1
        else:
            self.stats['total_legit'] += 1
        self.stats['recent_latency_ms'] = elapsed_ms

        # Extract top 3 SHAP factors safely
        top_factors = decision_res.get('explanation', {}).get('top_factors', {})
        tf_keys = list(top_factors.keys())
        tf_vals = list(top_factors.values())

        # Construct 28-field SSE / API Event Structure
        event_payload = {
            # Core Transaction Info (1-10)
            'transaction_id': tx_id,
            'timestamp': raw_record.get('timestamp', datetime.now().isoformat()),
            'amount': amt,
            'card1': sender,
            'card2': receiver,
            'addr1': raw_record.get('addr1', 'N/A'),
            'ProductCD': raw_record.get('ProductCD', 'W'),
            'DeviceType': raw_record.get('DeviceType', 'desktop'),
            'P_emaildomain': raw_record.get('P_emaildomain', 'gmail.com'),
            'R_emaildomain': raw_record.get('R_emaildomain', 'gmail.com'),
            
            # Decision Results & Scores (11-17)
            'decision': decision_res['decision'],
            'action': decision_res['action'],
            'final_risk_score': decision_res['final_risk_score'],
            'rf_flagged': bool(decision_res['rf_flagged']),
            'supervised_score': decision_res['component_scores']['supervised'],
            'anomaly_score': decision_res['component_scores']['anomaly'],
            'graph_risk_score': decision_res['component_scores']['graph'],
            
            # Top-3 SHAP Factors (18-23)
            'top_factor_1_name': tf_keys[0] if len(tf_keys) > 0 else 'TransactionAmt',
            'top_factor_1_value': tf_vals[0] if len(tf_vals) > 0 else 0.5,
            'top_factor_2_name': tf_keys[1] if len(tf_keys) > 1 else 'amount_deviation_score',
            'top_factor_2_value': tf_vals[1] if len(tf_vals) > 1 else 0.3,
            'top_factor_3_name': tf_keys[2] if len(tf_keys) > 2 else 'transaction_freq_24h',
            'top_factor_3_value': tf_vals[2] if len(tf_vals) > 2 else 0.1,
            
            # Streaming & Graph Metadata (24-28)
            'latency_ms': elapsed_ms,
            'source': source,
            'is_anomaly': norm_anomaly_score > 0.5,
            'hub_node_flag': hub_flag,
            'cycle_flag': cycle_flag
        }

        # Broadcast payload to active SSE client connections
        self.broadcast_sse(event_payload)
        return event_payload

if __name__ == "__main__":
    pipeline = FraudStreamingPipeline()
    res = pipeline.process_transaction(pipeline.generator.generate_transaction())
    print(f"Processed 28-field event in {res['latency_ms']} ms. Decision: {res['decision']}")
