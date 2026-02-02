"""
Utility functions for the skin cancer detection project.
"""

import os
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import LOGS_DIR, TRAINING_LOG_PATH


def setup_logging(log_file=None):
    """
    Setup logging configuration.
    
    Args:
        log_file (str): Path to the log file. If None, uses TRAINING_LOG_PATH
    """
    if log_file is None:
        log_file = TRAINING_LOG_PATH
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_directories():
    """
    Create all necessary directories for the project.
    """
    from src.config import (
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        TRAIN_DIR, VAL_DIR, TEST_DIR,
        MODELS_DIR, CHECKPOINT_PATH,
        LOGS_DIR, TENSORBOARD_LOG_DIR
    )
    
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        TRAIN_DIR, VAL_DIR, TEST_DIR,
        MODELS_DIR, CHECKPOINT_PATH,
        LOGS_DIR, TENSORBOARD_LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logging.info("All directories created successfully")


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history object
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_model_info(model, filepath):
    """
    Save model information to a JSON file.
    
    Args:
        model: Keras model
        filepath (str): Path to save the JSON file
    """
    model_info = {
        'total_params': int(model.count_params()),
        'trainable_params': int(sum([w.shape.num_elements() for w in model.trainable_weights])),
        'layers': len(model.layers),
        'input_shape': [int(dim) for dim in model.input_shape[1:]],
        'output_shape': [int(dim) for dim in model.output_shape[1:]]
    }
    
    with open(filepath, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    logging.info(f"Model info saved to: {filepath}")


def count_dataset_images(data_dir):
    """
    Count the number of images in each class of a dataset directory.
    
    Args:
        data_dir (Path or str): Path to the dataset directory
        
    Returns:
        dict: Dictionary with class names as keys and image counts as values
    """
    data_dir = Path(data_dir)
    counts = {}
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            count = sum(1 for f in class_dir.iterdir() 
                       if f.suffix.lower() in image_extensions)
            counts[class_dir.name] = count
    
    return counts
