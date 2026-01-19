"""
Evaluation and visualization functions for HAR Classification.

This module provides functions for:
- Creating confusion matrices
- Visualizing feature importance
- Generating comprehensive evaluation plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

def wrap_labels(labels):
    """Split labels on underscores to wrap long labels across multiple lines."""
    return [label.replace('_', '_\n') for label in labels]

def feature_importance_plot(model, X_train, save_path=None, top_n=20):
    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    # Sort from most to least important
    feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    # Top N most important features
    top_features = feature_imp_df.head(top_n) 

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()  # most important at top
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'results', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    


def activity_distribution_plot(df, save_path=None):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Activity')
    plt.title('Activity Distribution')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    
    # Wrap long activity labels across multiple lines
    activity_labels = df['Activity'].unique()
    wrapped_labels = wrap_labels(activity_labels)
    ax.set_xticklabels(wrapped_labels)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'results', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    

def confusion_matrix_plot(y_test_encoded, y_test_pred, encoder, save_path=None):
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    
    wrapped_labels = wrap_labels(encoder.classes_)

    plt.figure(figsize=(10, 8))
    # create heatmap of confusion matrix 
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=wrapped_labels,
        yticklabels=wrapped_labels,
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'results', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

