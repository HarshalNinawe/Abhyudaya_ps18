"""
Dataset loading and preprocessing utilities.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, PREPROCESSING_PARAMS
)


def create_data_generators():
    """
    Create data generators for training, validation, and testing.
    
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        **PREPROCESSING_PARAMS
    )
    
    # Validation and test data generators (only rescaling)
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array
