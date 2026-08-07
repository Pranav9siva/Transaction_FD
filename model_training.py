import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

# Try to import SMOTE, handle if not available (though requested)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imblearn not found. SMOTE will be skipped.")

def train_model(data_path, model_output_path="fraud_model.pkl", test_size=0.2, random_state=42):
    """
    Trains a fraud detection model using Random Forest and SMOTE for imbalance handling.
    
    Args:
        data_path (str): Path to the preprocessed CSV dataset.
        model_output_path (str): Path to save the trained model.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    print(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Identify target and features
    target_col = 'isFraud'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Class distribution before split:\n{y.value_counts(normalize=True)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Handle Class Imbalance using SMOTE
    if HAS_SMOTE:
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Resampled training shape: {X_train_resampled.shape}")
        print(f"Class distribution after SMOTE:\n{y_train_resampled.value_counts(normalize=True)}")
    else:
        print("Skipping SMOTE (library not available). Using original training data.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Train Random Forest
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_state, 
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    print("\n--- Model Evaluation ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    print(f"Saving model to {model_output_path}...")
    joblib.dump(clf, model_output_path)
    print("Model saved successfully.")
    
    return metrics

if __name__ == "__main__":
    import os
    default_data_path = "fraud_train_preprocessed.csv" if os.path.exists("fraud_train_preprocessed.csv") else r"c:\Users\sivap\Downloads\soft_eng\fraud_train_preprocessed.csv"
    
    try:
        train_model(default_data_path)
    except Exception as e:
        print(f"An error occurred: {e}")
