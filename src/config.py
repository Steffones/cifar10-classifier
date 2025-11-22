"""
Configuration file for CIFAR-10 Classification Project
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dataset parameters
IMAGE_SIZE = (96, 96)  # MobileNetV2 works well with 96x96
NUM_CLASSES = 10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Training parameters
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001

# Model parameters
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.0001

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'fill_mode': 'nearest'
}

# Callbacks parameters
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# Model save paths
MOBILENETV2_MODEL_PATH = os.path.join(MODELS_DIR, 'mobilenetv2_cifar10.h5')
CUSTOM_CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'custom_cnn_cifar10.h5')
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.json')

# Random seed for reproducibility
RANDOM_SEED = 42