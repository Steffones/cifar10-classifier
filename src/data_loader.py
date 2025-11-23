"""
Memory-efficient data loading and preprocessing for CIFAR-10 dataset
Uses tf.data.Dataset API for optimal RAM utilization and performance
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *

def load_cifar10_data():
    """
    Load CIFAR-10 dataset from Keras datasets
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_image(image, label, target_size=IMAGE_SIZE, is_training=True):
    """
    Preprocess a single image with optional data augmentation
    
    Args:
        image: Input image tensor
        label: Image label
        target_size: Target size tuple (height, width)
        is_training: Apply augmentation if True
    
    Returns:
        tuple: (preprocessed_image, label)
    """
    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Resize image
    image = tf.image.resize(image, target_size)
    
    # Data augmentation (only for training)
    if is_training:
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random saturation adjustment
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        
        # Clip values to [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

def create_tf_dataset(x, y, batch_size, target_size=IMAGE_SIZE, is_training=True, shuffle_buffer=10000):
    """
    Create optimized tf.data.Dataset pipeline
    
    Args:
        x: Input images numpy array
        y: Labels numpy array
        batch_size: Batch size
        target_size: Target image size
        is_training: Whether this is training data
        shuffle_buffer: Buffer size for shuffling
    
    Returns:
        tf.data.Dataset: Optimized dataset
    """
    # One-hot encode labels
    y_categorical = keras.utils.to_categorical(y, NUM_CLASSES)
    
    # Create dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((x, y_categorical))
    
    # Shuffle before preprocessing (for training only)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    
    # Map preprocessing function (parallel processing)
    dataset = dataset.map(
        lambda img, lbl: preprocess_image(img, lbl, target_size, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance (overlaps data preprocessing and model execution)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def create_data_generators(x_train, y_train, batch_size=BATCH_SIZE, 
                          target_size=IMAGE_SIZE, validation_split=VALIDATION_SPLIT):
    """
    Create memory-efficient training and validation datasets using tf.data API
    
    Args:
        x_train: Training images
        y_train: Training labels
        batch_size: Batch size
        target_size: Target image size
        validation_split: Fraction of data for validation
    
    Returns:
        tuple: (train_dataset, val_dataset, steps_per_epoch, validation_steps)
    """
    print("\n" + "="*70)
    print("CREATING OPTIMIZED DATA PIPELINE")
    print("="*70)
    print(f"Using tf.data.Dataset API for optimal performance")
    print(f"Target size: {target_size}")
    print(f"Batch size: {batch_size}")
    print(f"Validation split: {validation_split*100}%")
    
    # Calculate split index
    num_train_samples = len(x_train)
    num_val = int(num_train_samples * validation_split)
    
    # Split data
    x_val = x_train[:num_val]
    y_val = y_train[:num_val]
    x_train_split = x_train[num_val:]
    y_train_split = y_train[num_val:]
    
    print(f"\nTraining samples: {len(x_train_split)}")
    print(f"Validation samples: {len(x_val)}")
    
    # Create datasets
    train_dataset = create_tf_dataset(
        x_train_split, 
        y_train_split, 
        batch_size, 
        target_size, 
        is_training=True
    )
    
    val_dataset = create_tf_dataset(
        x_val, 
        y_val, 
        batch_size, 
        target_size, 
        is_training=False
    )
    
    # Calculate steps
    steps_per_epoch = len(x_train_split) // batch_size
    validation_steps = len(x_val) // batch_size
    
    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print("="*70)
    
    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def create_test_dataset(x_test, y_test, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE):
    """
    Create test dataset for evaluation
    
    Args:
        x_test: Test images
        y_test: Test labels
        batch_size: Batch size
        target_size: Target image size
    
    Returns:
        tf.data.Dataset: Test dataset
    """
    print(f"\nCreating test dataset...")
    print(f"Test samples: {len(x_test)}")
    
    test_dataset = create_tf_dataset(
        x_test,
        y_test,
        batch_size,
        target_size,
        is_training=False
    )
    
    return test_dataset

def preprocess_single_image(image, target_size=IMAGE_SIZE):
    """
    Preprocess a single image for prediction (used in Streamlit/Gradio)
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size (height, width)
    
    Returns:
        np.array: Preprocessed image ready for prediction
    """
    # Convert PIL to numpy if needed
    if hasattr(image, 'resize'):
        image = image.resize(target_size)
        image = np.array(image)
    
    # Ensure correct shape
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    
    # Resize if needed
    if image.shape[:2] != target_size:
        image = tf.image.resize(image, target_size).numpy()
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def get_class_distribution(y_data):
    """
    Get class distribution statistics
    
    Args:
        y_data: Labels (one-hot encoded or integer)
    
    Returns:
        dict: Class distribution
    """
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:
        # One-hot encoded
        class_counts = np.sum(y_data, axis=0)
    else:
        # Integer labels
        class_counts = np.bincount(y_data.flatten(), minlength=NUM_CLASSES)
    
    distribution = {CLASS_NAMES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)}
    return distribution

def estimate_memory_usage(num_samples, image_size, batch_size):
    """
    Estimate memory usage for dataset
    
    Args:
        num_samples: Number of samples
        image_size: Image size tuple
        batch_size: Batch size
    
    Returns:
        dict: Memory estimates
    """
    # Calculate memory per image (float32 = 4 bytes)
    bytes_per_image = image_size[0] * image_size[1] * 3 * 4  # 3 channels, 4 bytes per float32
    
    # Total memory for all images (in MB)
    total_memory_mb = (num_samples * bytes_per_image) / (1024 * 1024)
    
    # Memory per batch
    batch_memory_mb = (batch_size * bytes_per_image) / (1024 * 1024)
    
    return {
        'total_dataset_mb': total_memory_mb,
        'per_batch_mb': batch_memory_mb,
        'recommended_prefetch': min(10, max(2, int(1000 / batch_memory_mb)))
    }