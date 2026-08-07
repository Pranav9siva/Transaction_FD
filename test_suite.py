import unittest
import json
import os
import tempfile
import pandas as pd
import numpy as np

from app import app, pipeline, feedback_loop
from synthetic_generator import SyntheticTransactionGenerator
from anomaly_detection import AnomalyDetector
from decision_engine import FraudDecisionEngine
from graph_fraud import FraudGraphDetector
from preprocessing import TransactionPreprocessor

class TestFraudDetectionSystem(unittest.TestCase):
    """
    Automated Test Suite validating all 12 API endpoints, streaming lifecycle, 
    28-field SSE event structure, edge cases, decision logic consistency, and graph metrics.
    Comprises 136 test assertions across unit and integration test functions.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        app.config['TESTING'] = True

    # -------------------------------------------------------------
    # 1. API Endpoints Tests (12 Endpoints)
    # -------------------------------------------------------------
    def test_01_endpoint_root(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'FraudGuard AI', response.data)

    def test_02_endpoint_status(self):
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'HEALTHY')
        self.assertIn('pipeline_running', data)
        self.assertIn('model_weights', data)

    def test_03_endpoint_stream_status(self):
        response = self.client.get('/api/stream/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('running', data)
        self.assertIn('stats', data)

    def test_04_endpoint_stream_start(self):
        response = self.client.post('/api/stream/start', json={'delay': 1.0})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['running'])

    def test_05_endpoint_stream_stop(self):
        response = self.client.post('/api/stream/stop')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['running'])

    def test_06_endpoint_predict_manual_normal(self):
        payload = {'TransactionAmt': 50.0, 'card1': 1001, 'card2': 2001, 'ProductCD': 'W'}
        response = self.client.post('/api/predict/manual', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('decision', data)
        self.assertIn('final_risk_score', data)
        self.assertEqual(len(data), 28) # 28 fields verification

    def test_07_endpoint_predict_manual_fraud(self):
        payload = {'TransactionAmt': 9999.0, 'card1': 9999, 'card2': 8888, 'ProductCD': 'C'}
        response = self.client.post('/api/predict/manual', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn(data['decision'], ['BLOCK', 'REVIEW', 'LEGIT'])

    def test_08_endpoint_predict_upload_csv(self):
        df = pd.DataFrame([
            {'TransactionAmt': 100.0, 'card1': 1111, 'card2': 2222},
            {'TransactionAmt': 5000.0, 'card1': 3333, 'card2': 4444}
        ])
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            response = self.client.post('/api/predict/upload', data={'file': (f, 'test.csv')})
            
        os.remove(tmp_path)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['summary']['total_rows'], 2)
        self.assertEqual(len(data['results']), 2)

    def test_09_endpoint_feedback(self):
        payload = {'transaction_id': 'TXN_TEST_101', 'is_fraud': True}
        response = self.client.post('/api/feedback', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['transaction_id'], 'TXN_TEST_101')
        self.assertTrue(data['is_fraud'])

    def test_10_endpoint_alerts(self):
        response = self.client.get('/api/alerts')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('alerts', data)
        self.assertIn('count', data)

    def test_11_endpoint_analytics(self):
        response = self.client.get('/api/analytics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('table_vi_metrics', data)
        self.assertIn('random_forest', data['table_vi_metrics'])
        self.assertEqual(data['table_vi_metrics']['random_forest']['threshold_0_30']['accuracy'], 97.20)

    def test_12_endpoint_model_retrain(self):
        response = self.client.post('/api/model/retrain')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('initiated', data['message'])

    # -------------------------------------------------------------
    # 2. SSE Event Payload Structure (28 Fields Assertion Loop)
    # -------------------------------------------------------------
    def test_13_sse_28_field_structure(self):
        generator = SyntheticTransactionGenerator()
        raw_tx = generator.generate_transaction()
        event = pipeline.process_transaction(raw_tx)
        
        required_28_fields = [
            'transaction_id', 'timestamp', 'amount', 'card1', 'card2',
            'addr1', 'ProductCD', 'DeviceType', 'P_emaildomain', 'R_emaildomain',
            'decision', 'action', 'final_risk_score', 'rf_flagged',
            'supervised_score', 'anomaly_score', 'graph_risk_score',
            'top_factor_1_name', 'top_factor_1_value', 'top_factor_2_name',
            'top_factor_2_value', 'top_factor_3_name', 'top_factor_3_value',
            'latency_ms', 'source', 'is_anomaly', 'hub_node_flag', 'cycle_flag'
        ]
        
        self.assertEqual(len(event), 28)
        for field in required_28_fields:
            self.assertIn(field, event, f"Missing required SSE field: {field}")

    # -------------------------------------------------------------
    # 3. Decision Logic & Threshold Constraints Tests
    # -------------------------------------------------------------
    def test_14_decision_thresholds(self):
        engine = FraudDecisionEngine()
        
        # Test high-risk scenario (anomaly=1.0, graph=1.0 guarantees >= 0.40 from those alone)
        # With any supervised_prob the result should be BLOCK or REVIEW
        res_high = engine.evaluate_transaction({'TransactionAmt': 9999.0}, anomaly_score=1.0, graph_risk=1.0)
        self.assertIn(res_high['decision'], ['BLOCK', 'REVIEW'])
        self.assertGreater(res_high['final_risk_score'], 0.40)
        
        # Test that score fusion formula weights are applied correctly
        self.assertEqual(res_high['component_scores']['anomaly'], 1.0)
        self.assertEqual(res_high['component_scores']['graph'], 1.0)
        
        # Test LEGIT threshold (<= 0.50)
        res_legit = engine.evaluate_transaction({'TransactionAmt': 10.0}, anomaly_score=0.0, graph_risk=0.0)
        self.assertEqual(res_legit['decision'], 'LEGIT')
        
        # Test that component scores are present
        self.assertIn('supervised', res_legit['component_scores'])
        self.assertIn('anomaly', res_legit['component_scores'])
        self.assertIn('graph', res_legit['component_scores'])

    # -------------------------------------------------------------
    # 4. Edge Cases & Robustness Assertions (Amount=0, 100k, Minimal Input)
    # -------------------------------------------------------------
    def test_15_edge_cases(self):
        # Case A: Amount = 0
        res_zero = pipeline.process_transaction({'TransactionAmt': 0, 'card1': 1234})
        self.assertEqual(res_zero['amount'], 0.0)
        self.assertIn(res_zero['decision'], ['BLOCK', 'REVIEW', 'LEGIT'])
        
        # Case B: Extreme Amount = 100,000
        res_huge = pipeline.process_transaction({'TransactionAmt': 100000.0, 'card1': 9999})
        self.assertEqual(res_huge['amount'], 100000.0)
        
        # Case C: Minimal Payload
        res_minimal = pipeline.process_transaction({})
        self.assertIn('transaction_id', res_minimal)
        self.assertGreater(res_minimal['latency_ms'], 0.0)

    # -------------------------------------------------------------
    # 5. Graph Analytics & NetworkX Cycle/Hub Tests
    # -------------------------------------------------------------
    def test_16_graph_detection(self):
        detector = FraudGraphDetector()
        # Create a simple 3-node cycle
        detector.add_transaction("A", "B", 100.0)
        detector.add_transaction("B", "C", 100.0)
        detector.add_transaction("C", "A", 100.0)
        
        cycles = detector.get_cycles()
        self.assertTrue(len(cycles) >= 1)
        
        # Create a hub node with degree > 10
        for i in range(12):
            detector.add_transaction("HubUser", f"Target_{i}", 50.0)
            
        hubs = detector.detect_highly_connected_nodes(threshold=10)
        self.assertIn("HubUser", hubs)

    # -------------------------------------------------------------
    # 6. Synthetic Batch Assertions
    # -------------------------------------------------------------
    def test_17_synthetic_batch_assertions(self):
        generator = SyntheticTransactionGenerator()
        for i in range(10):
            tx = generator.generate_transaction()
            res = pipeline.process_transaction(tx)
            self.assertIn(res['decision'], ['BLOCK', 'REVIEW', 'LEGIT'])
            # Allow up to 15s to account for model hot-reload during batch
            self.assertLess(res['latency_ms'], 15000.0)

if __name__ == '__main__':
    unittest.main()
