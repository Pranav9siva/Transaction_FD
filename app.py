import os
import json
import queue
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

from streaming_pipeline import FraudStreamingPipeline
from feedback_loop import FeedbackLoop
from alert_system import AlertSystem

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

app = Flask(__name__, template_folder="templates", static_folder="static")
if HAS_CORS:
    CORS(app)

# Global Pipeline Instance
pipeline = FraudStreamingPipeline()
feedback_loop = FeedbackLoop()
alert_system = AlertSystem()

# Automatically start streaming on launch for demonstration
pipeline.start_pipeline(delay=2.0)

# -------------------------------------------------------------
# 12 REST API ENDPOINTS
# -------------------------------------------------------------

# 1. Dashboard Web UI Endpoint
@app.route('/')
def index():
    return render_template('index.html')

# 2. System Status Endpoint
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'HEALTHY',
        'system': 'Real-Time Transaction Fraud Detection System',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'pipeline_running': pipeline.running,
        'model_weights': {'w_s': pipeline.decision_engine.w_s, 'w_a': pipeline.decision_engine.w_a, 'w_g': pipeline.decision_engine.w_g},
        'rf_threshold': pipeline.decision_engine.rf_threshold
    })

# 3. SSE Live Events Stream Endpoint
@app.route('/api/stream/events', methods=['GET'])
def stream_events():
    def event_generator():
        client_queue = queue.Queue(maxsize=100)
        pipeline.add_sse_listener(client_queue)
        try:
            while True:
                try:
                    payload = client_queue.get(timeout=20.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    # Ping keep-alive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pipeline.remove_sse_listener(client_queue)

    return Response(event_generator(), mimetype='text/event-stream')

# 4. Start Streaming Endpoint
@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    data = request.get_json(silent=True) or {}
    delay = float(data.get('delay', 1.5))
    pipeline.start_pipeline(delay=delay)
    return jsonify({'message': 'Streaming pipeline started', 'running': pipeline.running, 'delay': delay})

# 5. Stop Streaming Endpoint
@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    pipeline.stop_pipeline()
    return jsonify({'message': 'Streaming pipeline stopped', 'running': pipeline.running})

# 6. Stream Status Endpoint
@app.route('/api/stream/status', methods=['GET'])
def stream_status():
    return jsonify({
        'running': pipeline.running,
        'stats': pipeline.stats,
        'active_sse_listeners': len(pipeline.sse_listeners)
    })

# 7. Manual Single Transaction Prediction Endpoint
@app.route('/api/predict/manual', methods=['POST'])
def predict_manual():
    record = request.get_json(force=True)
    if not record:
        return jsonify({'error': 'Invalid payload'}), 400
        
    result_event = pipeline.process_transaction(record, source="MANUAL")
    return jsonify(result_event)

# 8. CSV Batch Prediction Upload Endpoint (up to 1000 rows)
@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV format'}), 400
        
    try:
        df = pd.read_csv(file)
        if len(df) > 1000:
            df = df.head(1000)
            
        results = []
        for idx, row in df.iterrows():
            rec = row.to_dict()
            res = pipeline.process_transaction(rec, source="BATCH_CSV")
            results.append(res)
            
        summary = {
            'total_rows': len(results),
            'blocked_count': sum(1 for r in results if r['decision'] == 'BLOCK'),
            'review_count': sum(1 for r in results if r['decision'] == 'REVIEW'),
            'legit_count': sum(1 for r in results if r['decision'] == 'LEGIT'),
            'avg_risk_score': round(float(np.mean([r['final_risk_score'] for r in results])), 4) if results else 0.0,
            'avg_latency_ms': round(float(np.mean([r['latency_ms'] for r in results])), 2) if results else 0.0
        }
        
        return jsonify({
            'summary': summary,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': f"Failed to process CSV: {str(e)}"}), 500

# 9. Analyst Feedback Submission Endpoint
@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json(force=True)
    tx_id = data.get('transaction_id')
    is_fraud = data.get('is_fraud')
    
    if not tx_id or is_fraud is None:
        return jsonify({'error': 'Missing transaction_id or is_fraud indicator'}), 400
        
    feedback_loop.log_feedback(tx_id, bool(is_fraud))
    return jsonify({
        'message': 'Feedback successfully submitted',
        'transaction_id': tx_id,
        'is_fraud': bool(is_fraud)
    })

# 10. Historical Alerts Fetching Endpoint
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    log_file = "alerts.jsonl"
    alerts = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        alerts.append(json.loads(line.strip()))
                    except:
                        pass
    # Return last 50 alerts
    return jsonify({'count': len(alerts), 'alerts': alerts[-50:]})

# 11. Classifier & System Analytics Endpoint (Table VI matching)
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    table_vi = {
        'random_forest': {
            'threshold_0_30': {'accuracy': 97.20, 'precision': 77.91, 'recall': 35.38, 'f1_score': 0.4866, 'roc_auc': 0.9072},
            'threshold_0_50': {'accuracy': 96.61, 'precision': 54.76, 'recall': 55.40, 'f1_score': 0.5508, 'roc_auc': 0.9072}
        },
        'isolation_forest': {
            'accuracy': 92.24, 'precision': 12.31, 'recall': 17.49, 'f1_score': 0.1445, 'roc_auc': 0.7687, 'anomalies_flagged': '3,145 / 59,054 (5.33%)'
        },
        'combined_system': {
            'either_flags_or': {'precision': 26.64, 'recall': 59.60, 'f1_score': 0.3682},
            'both_flag_and': {'precision': 68.06, 'recall': 13.29, 'f1_score': 0.2223}
        }
    }
    
    return jsonify({
        'table_vi_metrics': table_vi,
        'live_stats': pipeline.stats,
        'graph_metrics': {
            'nodes_count': pipeline.graph_detector.graph.number_of_nodes(),
            'edges_count': pipeline.graph_detector.graph.number_of_edges(),
            'hubs': pipeline.graph_detector.detect_highly_connected_nodes(threshold=10),
            'cycles_count': len(pipeline.graph_detector.get_cycles())
        }
    })

# 12. Hot-Swap Model Retraining Trigger Endpoint
@app.route('/api/model/retrain', methods=['POST'])
def trigger_retrain():
    def retrain_async():
        feedback_loop.retrain()
        
    t = threading.Thread(target=retrain_async)
    t.start()
    return jsonify({'message': 'Hot-swap retraining process initiated in background'})

if __name__ == '__main__':
    print("[App] Starting Fraud Detection Web Application on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
