"""
Model architectures for CIFAR-10 classification
Optimized for memory efficiency and robust fine-tuning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from config import *

def build_mobilenetv2_model(input_shape=IMAGE_SIZE + (3,), trainable_base=BASE_MODEL_TRAINABLE_PHASE1):
    """
    Build MobileNetV2 model with transfer learning
    
    Args:
        input_shape: Input image shape (height, width, channels)
        trainable_base: Whether to make base model trainable
    
    Returns:
        keras.Model: MobileNetV2 model (not compiled)
    """
    print("\n" + "="*70)
    print("BUILDING MOBILENETV2 MODEL WITH TRANSFER LEARNING")
    print("="*70)
    
    # Load pre-trained MobileNetV2 (without top layers)
    print("Loading pre-trained MobileNetV2 from ImageNet...")
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Width multiplier
    )
    
    # Set base model trainability
    base_model.trainable = trainable_base
    
    print(f"Base model loaded: {len(base_model.layers)} layers")
    print(f"Base model trainable: {trainable_base}")
    
    # Build model using Functional API
    inputs = keras.Input(shape=input_shape, name='input_image')
    
    # Preprocess input for MobileNetV2 (scales to [-1, 1])
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model - set training=False for inference mode even if trainable=True
    # This keeps batch norm layers in inference mode during initial training
    x = base_model(x, training=False)
    
    # Global average pooling - reduces spatial dimensions to single vector
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Batch normalization for stability
    x = layers.BatchNormalization(name='bn_after_gap')(x)
    
    # Dropout for regularization
    x = layers.Dropout(DROPOUT_RATE, name='dropout_1')(x)
    
    # Dense layer with L2 regularization
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
        name='dense_256'
    )(x)
    
    # Batch normalization
    x = layers.BatchNormalization(name='bn_dense')(x)
    
    # Another dropout (lighter)
    x = layers.Dropout(DROPOUT_RATE * 0.5, name='dropout_2')(x)
    
    # Output layer - no regularization on final layer
    outputs = layers.Dense(
        NUM_CLASSES, 
        activation='softmax',
        name='predictions'
    )(x)
    
    # Create model
    model = keras.Model(inputs, outputs, name='MobileNetV2_CIFAR10')
    
    # Print summary
    print(f"\n{'='*70}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*70}")
    print(f"Input shape: {input_shape}")
    print(f"Base model: MobileNetV2 (ImageNet pretrained)")
    print(f"Total layers: {len(model.layers)}")
    print(f"Base model layers: {len(base_model.layers)}")
    print(f"Custom head layers: {len(model.layers) - len(base_model.layers)}")
    
    return model

def build_custom_cnn_model(input_shape=IMAGE_SIZE + (3,)):
    """
    Build custom CNN model from scratch
    
    Architecture:
    - 3 Convolutional blocks (Conv2D -> BatchNorm -> Conv2D -> MaxPool -> Dropout)
    - Global Average Pooling
    - 2 Dense layers with dropout
    - Softmax output
    
    Args:
        input_shape: Input image shape
    
    Returns:
        keras.Model: Custom CNN model (not compiled)
    """
    print("\n" + "="*70)
    print("BUILDING CUSTOM CNN MODEL FROM SCRATCH")
    print("="*70)
    
    model = models.Sequential(name='Custom_CNN_CIFAR10')
    
    # Block 1: 32 filters
    model.add(layers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        input_shape=input_shape,
        name='conv1_1'
    ))
    model.add(layers.BatchNormalization(name='bn1_1'))
    model.add(layers.Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        name='conv1_2'
    ))
    model.add(layers.BatchNormalization(name='bn1_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool1'))
    model.add(layers.Dropout(0.25, name='dropout1'))
    
    # Block 2: 64 filters
    model.add(layers.Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        name='conv2_1'
    ))
    model.add(layers.BatchNormalization(name='bn2_1'))
    model.add(layers.Conv2D(
        64, (3, 3),
        activation='relu',
        padding='same',
        name='conv2_2'
    ))
    model.add(layers.BatchNormalization(name='bn2_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool2'))
    model.add(layers.Dropout(0.25, name='dropout2'))
    
    # Block 3: 128 filters
    model.add(layers.Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        name='conv3_1'
    ))
    model.add(layers.BatchNormalization(name='bn3_1'))
    model.add(layers.Conv2D(
        128, (3, 3),
        activation='relu',
        padding='same',
        name='conv3_2'
    ))
    model.add(layers.BatchNormalization(name='bn3_2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool3'))
    model.add(layers.Dropout(0.25, name='dropout3'))
    
    # Global Average Pooling (better than Flatten for generalization)
    model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
    
    # Dense layers
    model.add(layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
        name='dense_512'
    ))
    model.add(layers.BatchNormalization(name='bn_dense1'))
    model.add(layers.Dropout(DROPOUT_RATE, name='dropout4'))
    
    model.add(layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
        name='dense_256'
    ))
    model.add(layers.BatchNormalization(name='bn_dense2'))
    model.add(layers.Dropout(DROPOUT_RATE * 0.5, name='dropout5'))
    
    # Output layer
    model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='predictions'))
    
    print(f"\nCustom CNN Architecture:")
    print(f"Total layers: {len(model.layers)}")
    print(f"Input shape: {input_shape}")
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE, metrics=None):
    """
    Compile model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        metrics: List of metrics (uses default if None)
    
    Returns:
        keras.Model: Compiled model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    
    # Adam optimizer with gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Gradient clipping for stability
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    print(f"\n{'='*70}")
    print("MODEL COMPILATION")
    print(f"{'='*70}")
    print(f"Optimizer: Adam")
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient clipping: 1.0")
    print(f"Loss: Categorical Crossentropy")
    print(f"Metrics: {', '.join([m if isinstance(m, str) else m.name for m in metrics])}")
    print(f"{'='*70}")
    
    return model

def unfreeze_model_layers(model, num_layers_to_unfreeze=UNFREEZE_LAYERS):
    """
    Unfreeze top layers of MobileNetV2 base model for fine-tuning
    
    Args:
        model: Keras model with MobileNetV2 base
        num_layers_to_unfreeze: Number of top layers to unfreeze
    
    Returns:
        keras.Model: Model with unfrozen layers
    """
    print(f"\n{'='*70}")
    print("UNFREEZING LAYERS FOR FINE-TUNING")
    print(f"{'='*70}")
    
    # Find MobileNetV2 base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and 'mobilenet' in layer.name.lower():
            base_model = layer
            break
    
    # If not found by type, try by name
    if base_model is None:
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                base_model = layer
                break
    
    if base_model is None:
        print("⚠️ WARNING: Could not find MobileNetV2 base model")
        print("Skipping fine-tuning. Model structure:")
        for i, layer in enumerate(model.layers):
            print(f"  Layer {i}: {layer.name} (type: {type(layer).__name__})")
        return model
    
    print(f"Found base model: {base_model.name}")
    print(f"Base model has {len(base_model.layers)} layers")
    
    # Make base model trainable
    base_model.trainable = True
    
    # Freeze batch normalization layers (important for fine-tuning)
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    # Unfreeze only the last N layers
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers_to_unfreeze)
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # Skip BatchNorm layers even in unfrozen section
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    
    # Count trainable layers
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    frozen_count = len(base_model.layers) - trainable_count
    
    print(f"\nFine-tuning configuration:")
    print(f"  Layers to unfreeze: {num_layers_to_unfreeze}")
    print(f"  Frozen layers: {frozen_count}")
    print(f"  Trainable layers: {trainable_count}")
    print(f"  Total base model layers: {total_layers}")
    print(f"{'='*70}")
    
    return model

def print_model_summary(model, show_trainable=True):
    """
    Print detailed model summary with parameter counts
    
    Args:
        model: Keras model
        show_trainable: Show trainable status of each layer
    """
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    # Standard Keras summary
    model.summary()
    
    # Detailed parameter count
    trainable_params = sum([
        tf.keras.backend.count_params(w) 
        for w in model.trainable_weights
    ])
    non_trainable_params = sum([
        tf.keras.backend.count_params(w) 
        for w in model.non_trainable_weights
    ])
    total_params = trainable_params + non_trainable_params
    
    print(f"\n{'='*70}")
    print("PARAMETER STATISTICS")
    print(f"{'='*70}")
    print(f"Total parameters:        {total_params:>15,}")
    print(f"Trainable parameters:    {trainable_params:>15,}")
    print(f"Non-trainable parameters:{non_trainable_params:>15,}")
    print(f"Trainable percentage:    {100*trainable_params/total_params:>14.2f}%")
    
    # Model size estimation
    model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
    print(f"Estimated model size:    {model_size_mb:>14.2f} MB")
    print(f"{'='*70}")
    
    # Show trainable status if requested
    if show_trainable:
        print(f"\nTRAINABLE LAYER SUMMARY")
        print(f"{'='*70}")
        trainable_layers = [layer for layer in model.layers if layer.trainable]
        frozen_layers = [layer for layer in model.layers if not layer.trainable]
        
        print(f"Trainable layers: {len(trainable_layers)}")
        print(f"Frozen layers: {len(frozen_layers)}")
        
        if len(trainable_layers) < 20:  # Only show if not too many
            print("\nTrainable layers:")
            for layer in trainable_layers:
                params = layer.count_params()
                print(f"  • {layer.name:<30} {params:>10,} params")

def get_model_config(model):
    """
    Get model configuration for logging
    
    Args:
        model: Keras model
    
    Returns:
        dict: Model configuration
    """
    trainable_params = sum([
        tf.keras.backend.count_params(w) 
        for w in model.trainable_weights
    ])
    total_params = trainable_params + sum([
        tf.keras.backend.count_params(w) 
        for w in model.non_trainable_weights
    ])
    
    return {
        'model_name': model.name,
        'total_layers': len(model.layers),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }