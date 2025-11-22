"""
Model architectures for CIFAR-10 classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from config import *

def build_mobilenetv2_model(input_shape=(96, 96, 3), trainable_base=False):
    """
    Build MobileNetV2 model with transfer learning
    
    Args:
        input_shape: Input image shape
        trainable_base: Whether to make base model trainable
    
    Returns:
        keras.Model: Compiled MobileNetV2 model
    """
    print("\n" + "="*50)
    print("Building MobileNetV2 Model with Transfer Learning")
    print("="*50)
    
    # Load pre-trained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = trainable_base
    
    # Build custom classification head
    inputs = keras.Input(shape=input_shape)
    
    # Preprocess input for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Dense layer with L2 regularization
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION)
    )(x)
    
    # Another dropout
    x = layers.Dropout(DROPOUT_RATE * 0.5)(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs, outputs, name='MobileNetV2_CIFAR10')
    
    print(f"\nBase Model: MobileNetV2")
    print(f"Base Model Trainable: {trainable_base}")
    print(f"Total Layers: {len(model.layers)}")
    
    return model

def build_custom_cnn_model(input_shape=(96, 96, 3)):
    """
    Build custom CNN model from scratch
    
    Architecture:
    - 3 Convolutional blocks (Conv2D -> ReLU -> MaxPooling)
    - Flatten
    - 2 Dense layers
    - Dropout for regularization
    
    Args:
        input_shape: Input image shape
    
    Returns:
        keras.Model: Compiled custom CNN model
    """
    print("\n" + "="*50)
    print("Building Custom CNN Model")
    print("="*50)
    
    model = models.Sequential(name='Custom_CNN_CIFAR10')
    
    # Block 1: Convolutional Layer
    model.add(layers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        input_shape=input_shape
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2: Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3: Convolutional Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Flatten
    model.add(layers.Flatten())
    
    # Dense layers
    model.add(layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION)
    ))
    model.add(layers.Dropout(DROPOUT_RATE))
    
    model.add(layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION)
    ))
    model.add(layers.Dropout(DROPOUT_RATE * 0.5))
    
    # Output layer
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    
    print(f"\nCustom CNN Architecture:")
    print(f"Total Layers: {len(model.layers)}")
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        keras.Model: Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"\nModel compiled with:")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    print(f"  Loss: Categorical Crossentropy")
    print(f"  Metrics: Accuracy, Top-3 Accuracy")
    
    return model

def unfreeze_model_layers(model, num_layers_to_unfreeze=20):
    """
    Unfreeze top layers of the model for fine-tuning
    
    Args:
        model: Keras model
        num_layers_to_unfreeze: Number of layers to unfreeze from the top
    
    Returns:
        keras.Model: Model with unfrozen layers
    """
    print(f"\nUnfreezing top {num_layers_to_unfreeze} layers for fine-tuning...")
    
    # Get the base model (MobileNetV2)
    base_model = model.layers[2]  # Assuming base model is at index 2
    
    # Freeze all layers first
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers_to_unfreeze
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Trainable layers in base model: {trainable_count}/{len(base_model.layers)}")
    
    return model

def print_model_summary(model):
    """
    Print detailed model summary
    
    Args:
        model: Keras model
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    model.summary()
    
    # Count parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\nTotal parameters: {trainable_params + non_trainable_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("="*50)