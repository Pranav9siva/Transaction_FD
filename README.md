# Real-Time Transaction Fraud Detection System

This project is a Real-Time Transaction Fraud Detection System that leverages Machine Learning and Stream Processing to identify and flag fraudulent transactions as they occur.

## Project Explanation

### 1. Data 
The system ingests streaming transaction data (simulated as real-time). The transactions undergo complex preprocessing which generates over 260 features, including behavioral aggregations, standard encodings, and numerical imputations. The dataset is highly imbalanced (approx. 96.5% legitimate, 3.5% fraud), making fraud detection challenging.

### 2. Models
The detection engine employs a Hybrid Multi-Layered approach:
- **Supervised Random Forest with SMOTE**: A Random Forest classifier trained on the transaction dataset. To handle the class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is used to generate synthetic fraud samples during training, balancing the classes 50/50 and vastly improving the model's recall rate.
- **Unsupervised Isolation Forest**: An anomaly detection algorithm (set with a 3% contamination rate) that detects mathematically anomalous patterns outside the scope of known fraud.
- **Graph-Based Fraud Detection (NetworkX)**: Graph analytics that build a network of Sender/Receiver nodes to detect suspicious topologies like multi-hop cyclical transactions and high-degree fraud hubs.

### 3. Output
The system acts as a high-throughput pipeline. For every ingested transaction, it generates a robust **28-field JSON payload** containing:
- Core transaction metrics (Amount, Sender, Receiver).
- Multi-Model Scores (Supervised Score, Anomaly Score, Graph Risk).
- SHAP Feature Contributions (Top 3 factors driving the decision for XAI transparency).
- The final classification decision (`BLOCK`, `REVIEW`, or `LEGIT`).
This payload is streamed directly to the frontend dashboard using Server-Sent Events (SSE).

## Performance Metrics

After optimizing the model using the SMOTE algorithm on the dataset (testing on >100,000 transactions), the Random Forest model achieved the following real-world metrics:

- **Accuracy**: 98.0%
- **Precision**: 88.72% (When the model flags fraud, it is highly accurate, producing only 243 false alarms)
- **Recall**: 51.82% (The model successfully caught 1,912 fraud attacks, significantly improved via SMOTE)
- **F1 Score**: 65.42%

## Project Structure

- `app.py`: The Flask REST API and web application server.
- `streaming_pipeline.py`: Orchestrates the real-time processing pipeline.
- `decision_engine.py`: Fuses scores from multiple models and applies decision thresholds.
- `anomaly_detection.py`: Implementation of the Unsupervised Anomaly Detector.
- `graph_fraud.py`: Graph analytics module for detecting network-based fraud features.
- `synthetic_generator.py`: Generates realistic synthetic transaction streams.
- `feedback_loop.py`: Manages manual reviews and the automated model retraining loop.
- `templates/index.html`: The frontend Dashboard (featuring Dark/Light Mode and UI updates).

## Setup & Execution

### Requirements
Dependencies are listed in `requirements.txt`.
- Python 3.8+
- Scikit-Learn
- Pandas
- Numpy
- Flask
- SHAP
- NetworkX
- Imbalanced-Learn

### Running the System
1. Initialize a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Start the Streaming Pipeline & Web Dashboard:
   ```bash
   python app.py
   ```
3. Access the dashboard via your browser at `http://localhost:5000`.

## API Endpoints Overview
The Flask backend serves 12 endpoints. Key routes include:
- `GET /api/status`: System health and configuration.
- `POST /api/predict/manual`: Submit a single transaction for manual prediction.
- `GET /api/alerts`: Retrieve the latest fraud alerts.
- `GET /api/stream`: Server-Sent Events (SSE) endpoint for real-time dashboard updates.

## Design Philosophy

The decision logic is based on a weighted fusion formula:

$$ R_{final} = 0.6 \cdot P_{RF} + 0.2 \cdot S_{anomaly} + 0.2 \cdot S_{graph} $$

Transactions are then classified as:
- **BLOCK**: $R_{final} > 0.8$
- **REVIEW**: $0.5 < R_{final} \leq 0.8$
- **LEGIT**: $R_{final} \leq 0.5$
