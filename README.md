# Real-Time Transaction Fraud Detection System

This project is a Real-Time Transaction Fraud Detection System that leverages Machine Learning and Stream Processing to identify and flag fraudulent transactions as they occur.

## Key Features

- **Machine Learning Models**: Employs a hybrid approach utilizing a Supervised Random Forest Classifier for known fraud patterns and an Unsupervised Isolation Forest for anomaly detection.
- **Explainable AI (XAI)**: Integrates SHAP (SHapley Additive exPlanations) to provide feature-level explanations for fraud decisions, ensuring transparency.
- **Streaming Pipeline**: Simulates real-time data ingestion using a producer-consumer architecture to process high-throughput transaction streams with low latency.
- **Graph-Based Fraud Detection**: Utilizes NetworkX to perform graph analytics, identifying suspicious network patterns such as cyclical transactions and highly connected hubs.
- **Dynamic Feedback Loop**: Incorporates an automated retraining mechanism that updates the models based on human feedback (e.g., reviewed transactions).
- **RESTful API**: A Flask-based backend serving 12 dedicated endpoints for real-time inference, model management, alert monitoring, and system status.
- **Interactive Dashboard**: A modern, responsive web interface built with HTML/CSS/JS (featuring a glassmorphism design and Chart.js) to visualize real-time metrics, recent alerts, and system health.

## Project Structure

- `app.py`: The Flask REST API and web application server.
- `streaming_pipeline.py`: Orchestrates the real-time processing pipeline.
- `decision_engine.py`: Fuses scores from multiple models and applies decision thresholds.
- `anomaly_detection.py`: Implementation of the Unsupervised Anomaly Detector.
- `graph_fraud.py`: Graph analytics module for detecting network-based fraud features.
- `synthetic_generator.py`: Generates realistic synthetic transaction streams.
- `feedback_loop.py`: Manages manual reviews and the automated model retraining loop.
- `templates/index.html`: The frontend Dashboard.

## Setup & Execution

### Requirements
- Python 3.8+
- Scikit-Learn
- Pandas
- Numpy
- Flask
- SHAP
- NetworkX

### Running the System
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streaming Pipeline & Web Dashboard:
   ```bash
   python app.py
   ```
3. Access the dashboard via your browser at `http://localhost:5000`.
4. The backend API is available under `http://localhost:5000/api/*`.

## API Endpoints Overview

- `GET /api/status`: System health and configuration.
- `POST /api/predict/manual`: Submit a single transaction for manual prediction.
- `GET /api/alerts`: Retrieve the latest fraud alerts.
- `GET /api/stream`: Server-Sent Events (SSE) endpoint for real-time dashboard updates.

## Design Philosophy

This system aligns with state-of-the-art research in streaming fraud detection, emphasizing high throughput, multi-layered decision logic, and human-in-the-loop review processes. The decision logic is based on a weighted fusion formula:

$$ R_{final} = 0.6 \cdot P_{RF} + 0.2 \cdot S_{anomaly} + 0.2 \cdot S_{graph} $$

Transactions are then classified as:
- **BLOCK**: $R_{final} > 0.8$
- **REVIEW**: $0.5 < R_{final} \leq 0.8$
- **LEGIT**: $R_{final} \leq 0.5$
