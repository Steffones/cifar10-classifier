"""
Data loading and preprocessing for CIFAR-10 dataset
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
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

def preprocess_data(x_train, y_train, x_test, y_test, target_size=IMAGE_SIZE):
    """
    Preprocess CIFAR-10 data:
    - Normalize pixel values
    - Resize images
    - Convert labels to categorical
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        target_size: Target image size (height, width)
    
    Returns:
        tuple: Preprocessed (x_train, y_train, x_test, y_test)
    """
    print("\nPreprocessing data...")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Resize images if needed
    if target_size != (32, 32):
        print(f"Resizing images to {target_size}...")
        x_train = resize_images(x_train, target_size)
        x_test = resize_images(x_test, target_size)
    
    # Convert labels to categorical (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

def resize_images(images, target_size):
    """
    Resize images using OpenCV
    
    Args:
        images: Array of images
        target_size: Target size (height, width)
    
    Returns:
        np.array: Resized images
    """
    resized = np.zeros((len(images), target_size[0], target_size[1], 3))
    for i, img in enumerate(images):
        resized[i] = cv2.resize(img, target_size)
    return resized

def create_data_generators(x_train, y_train, batch_size=BATCH_SIZE):
    """
    Create data generators with augmentation for training
    
    Args:
        x_train: Training images
        y_train: Training labels
        batch_size: Batch size
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    print("\nCreating data generators with augmentation...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode'],
        validation_split=VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=batch_size,
        subset='training',
        shuffle=True
    )
    
    # Validation generator (no augmentation)
    validation_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=batch_size,
        subset='validation',
        shuffle=False
    )
    
    print(f"Training batches: {len(train_generator)}")
    print(f"Validation batches: {len(validation_generator)}")
    
    return train_generator, validation_generator

def preprocess_single_image(image, target_size=IMAGE_SIZE):
    """
    Preprocess a single image for prediction
    
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
    
    # Resize if needed
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def get_class_distribution(y_data):
    """
    Get class distribution statistics
    
    Args:
        y_data: Labels (one-hot encoded)
    
    Returns:
        dict: Class distribution
    """
    class_counts = np.sum(y_data, axis=0)
    distribution = {CLASS_NAMES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)}
    return distribution