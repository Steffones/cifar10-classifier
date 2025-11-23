"""
Configuration file for CIFAR-10 Classification Project
Optimized for systems with limited RAM (6-8GB)
"""

import os
import multiprocessing

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
ORIGINAL_IMAGE_SIZE = (32, 32)  # CIFAR-10 original size
NUM_CLASSES = 10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Training parameters - OPTIMIZED FOR 6GB RAM
BATCH_SIZE = 64  # Safe for 6GB RAM
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001

# Performance optimization
NUM_WORKERS = min(multiprocessing.cpu_count(), 4)  # Limit workers to avoid RAM overload
PREFETCH_BUFFER = 2  # Number of batches to prefetch

# Model parameters
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.0001

# Data augmentation parameters - SIMPLIFIED FOR PERFORMANCE
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
MIN_LR = 1e-7  # Minimum learning rate

# Model save paths
MOBILENETV2_MODEL_PATH = os.path.join(MODELS_DIR, 'mobilenetv2_cifar10.h5')
MOBILENETV2_KERAS_PATH = os.path.join(MODELS_DIR, 'mobilenetv2_cifar10.keras')  # Native Keras format
CUSTOM_CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'custom_cnn_cifar10.h5')
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, 'training_history.json')

# Random seed for reproducibility
RANDOM_SEED = 42

# Memory management settings
ALLOW_GROWTH = True  # Allow TensorFlow to allocate GPU memory as needed
MEMORY_LIMIT_MB = None  # Set to a value (e.g., 4096) to limit TensorFlow memory usage

# Training phases
PHASE_1_EPOCHS = 10  # Feature extraction phase
PHASE_2_EPOCHS = 10  # Fine-tuning phase

# MobileNetV2 fine-tuning settings
UNFREEZE_LAYERS = 30  # Number of layers to unfreeze for fine-tuning
BASE_MODEL_TRAINABLE_PHASE1 = False  # Keep base frozen in phase 1
BASE_MODEL_TRAINABLE_PHASE2 = True   # Unfreeze in phase 2

# Logging
VERBOSE = 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Performance monitoring
PROFILE_BATCH = 0  # Set to batch number to profile (0 = disabled)