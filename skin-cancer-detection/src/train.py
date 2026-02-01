"""
Training script for the skin cancer detection model.
"""

import os
import logging
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from src.config import (
    CHECKPOINT_PATH, BEST_MODEL_PATH,
    TENSORBOARD_LOG_DIR, TRAINING_LOG_PATH, EPOCHS
)
from src.dataset import create_data_generators
from src.model import create_cnn_model, create_transfer_learning_model
from src.utils import setup_logging


def get_callbacks():
    """
    Create callbacks for model training.
    
    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(TENSORBOARD_LOG_DIR),
            histogram_freq=1
        )
    ]
    
    return callbacks


def train_model(model_type='cnn', epochs=EPOCHS):
    """
    Train the skin cancer detection model.
    
    Args:
        model_type (str): Type of model to train ('cnn' or 'transfer')
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, history)
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with {model_type} model...")
    
    # Create data generators
    train_gen, val_gen, _ = create_data_generators()
    
    # Create model
    if model_type == 'cnn':
        model = create_cnn_model()
    elif model_type == 'transfer':
        model = create_transfer_learning_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info("Model architecture:")
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {BEST_MODEL_PATH}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train skin cancer detection model')
    parser.add_argument(
        '--model-type',
        type=str,
        default='cnn',
        choices=['cnn', 'transfer'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of training epochs'
    )
    
    args = parser.parse_args()
    
    train_model(model_type=args.model_type, epochs=args.epochs)
