"""
Utility functions for CIFAR-10 classification project
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from config import *

def save_training_history(history, filepath=TRAINING_HISTORY_PATH):
    """
    Save training history to JSON file
    
    Args:
        history: Training history object
        filepath: Path to save JSON file
    """
    history_dict = history.history
    
    # Convert numpy arrays to lists for JSON serialization
    for key in history_dict:
        history_dict[key] = [float(x) for x in history_dict[key]]
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"\n✓ Training history saved to {filepath}")

def load_training_history(filepath=TRAINING_HISTORY_PATH):
    """
    Load training history from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: Training history
    """
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves
    
    Args:
        history: Training history (dict or History object)
        save_path: Path to save plot (optional)
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - CIFAR-10 Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_sample_predictions(model, x_test, y_test, num_samples=12):
    """
    Plot sample predictions with true and predicted labels
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        num_samples: Number of samples to display
    """
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Make predictions
    predictions = model.predict(x_test[indices])
    
    # Create subplot grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Get image and labels
        img = x_test[idx]
        true_label = CLASS_NAMES[np.argmax(y_test[idx])]
        pred_label = CLASS_NAMES[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        
        # Plot image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Set title with color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)',
            fontsize=10,
            color=color,
            fontweight='bold'
        )
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y_data, title='Class Distribution'):
    """
    Plot class distribution bar chart
    
    Args:
        y_data: Labels (one-hot encoded)
        title: Plot title
    """
    class_counts = np.sum(y_data, axis=0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CLASS_NAMES, class_counts, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_callbacks(model_save_path):
    """
    Create training callbacks
    
    Args:
        model_save_path: Path to save best model
    
    Returns:
        list: List of callbacks
    """
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(RESULTS_DIR, 'logs'),
            histogram_freq=1
        )
    ]
    
    return callbacks

def get_model_size(model_path):
    """
    Get model file size in MB
    
    Args:
        model_path: Path to model file
    
    Returns:
        float: Model size in MB
    """
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def print_evaluation_summary(results):
    """
    Print formatted evaluation summary
    
    Args:
        results: Dictionary of evaluation results
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("="*60)