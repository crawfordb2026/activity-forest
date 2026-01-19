"""
Training script for Human Activity Recognition (HAR) Classification.

This script loads data, preprocesses it, trains a Random Forest model,
and evaluates performance on test set.

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from evaluate import feature_importance_plot, activity_distribution_plot, confusion_matrix_plot
import joblib
import os

# Best hyperparameters (from hyperparameter tuning)
BEST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'random_state': 42
}

def load_data(train_path='../data/train.csv', test_path='../data/test.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    # filter out both activity and subject cols
    X_train = train_df.drop(columns=['subject', 'Activity'])
    X_test = test_df.drop(columns=['subject', 'Activity'])

    # store only activity column for validation
    y_train = train_df['Activity']
    y_test = test_df['Activity']

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Z-score all features
    X_train_scaled = scaler.transform(X_train) # fit to training data only
    X_test_scaled = scaler.transform(X_test)

    encoder = LabelEncoder()
    encoder.fit(y_train) # fit to training data only

    # create encoded matricies
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    return X_train, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, encoder, scaler


def train_model(X_train, y_train, params=None):
    if params is None:
        params = BEST_PARAMS
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
    


def evaluate_model(model, X_test, y_test, encoder):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n")
    print(f"Test Set Accuracy: {accuracy*100:.2f}%")
    print(f"\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=encoder.classes_
    ))
    
    return y_pred, accuracy


def main():
    # Load data
    train_df, test_df = load_data()

    # Preprocess data
    X_train, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, encoder, scaler = preprocess_data(
        train_df, test_df
    )
    
    # Train model
    model = train_model(X_train_scaled, y_train_encoded)
    
    # Evaluate on test set
    y_pred, accuracy = evaluate_model(model, X_test_scaled, y_test_encoded, encoder)
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/final_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(encoder, '../models/encoder.pkl')
    print("\nModel saved to models/")

    feature_importance_plot(model, X_train, '../results/feature_importance.png')
    activity_distribution_plot(train_df, '../results/activity_distribution.png')
    confusion_matrix_plot(y_test_encoded, y_pred, encoder, '../results/confusion_matrix_heat_map.png')


    return model, scaler, encoder, accuracy, y_pred

if __name__ == "__main__":
    model, scaler, encoder, accuracy, y_pred = main()
