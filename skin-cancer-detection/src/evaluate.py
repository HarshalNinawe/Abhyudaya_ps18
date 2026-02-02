"""
Model evaluation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from tensorflow.keras.models import load_model
from src.config import BEST_MODEL_PATH
from src.dataset import create_data_generators
from src.utils import setup_logging
import logging


def evaluate_model(model_path=None):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path (str): Path to the saved model. If None, uses BEST_MODEL_PATH
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load model
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Get test data
    _, _, test_gen = create_data_generators()
    
    # Evaluate
    logger.info("Evaluating model on test data...")
    test_loss, test_accuracy, test_auc = model.evaluate(test_gen)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(
        y_true, y_pred,
        target_names=['Benign', 'Malignant']
    )
    logger.info(f"\n{report}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'predictions': predictions,
        'y_true': y_true,
        'y_pred': y_pred
    }


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path (str): Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate skin cancer detection model')
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the saved model'
    )
    
    args = parser.parse_args()
    
    results = evaluate_model(model_path=args.model_path)
    
    # Plot results
    plot_confusion_matrix(results['y_true'], results['y_pred'])
    plot_roc_curve(results['y_true'], results['predictions'])
