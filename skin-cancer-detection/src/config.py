"""
Configuration file for the skin cancer detection project.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
TRAIN_DIR = RAW_DATA_DIR / "train"
VAL_DIR = RAW_DATA_DIR / "val"
TEST_DIR = RAW_DATA_DIR / "test"

# Model hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Model configuration
NUM_CLASSES = 2  # benign and malignant
INPUT_SHAPE = (*IMAGE_SIZE, 3)

# Training configuration
CHECKPOINT_PATH = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "best_model.h5"
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
TRAINING_LOG_PATH = LOGS_DIR / "training.log"

# Preprocessing
PREPROCESSING_PARAMS = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "zoom_range": 0.2,
    "rescale": 1.0 / 255.0
}
