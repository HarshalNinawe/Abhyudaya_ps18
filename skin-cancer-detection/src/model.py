"""
Model architecture definition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from src.config import INPUT_SHAPE, LEARNING_RATE


def create_cnn_model():
    """
    Create a custom CNN model for skin cancer detection.
    
    Returns:
        keras.Model: Compiled model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model


def create_transfer_learning_model():
    """
    Create a transfer learning model using MobileNetV2.
    
    Returns:
        keras.Model: Compiled model
    """
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    return model


def unfreeze_base_model(model, layers_to_unfreeze=20):
    """
    Unfreeze the last N layers of the base model for fine-tuning.
    
    Args:
        model: The model to unfreeze layers in
        layers_to_unfreeze (int): Number of layers to unfreeze from the end
    """
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last N
    for layer in base_model.layers[:-layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    return model
