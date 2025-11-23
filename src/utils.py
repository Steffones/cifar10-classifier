"""
Utility functions for CIFAR-10 classification project
Enhanced with better error handling and monitoring
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
from datetime import datetime
from config import *

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

def save_training_history(history, filepath=TRAINING_HISTORY_PATH):
    """
    Save training history to JSON file with metadata
    
    Args:
        history: Training history object or dict
        filepath: Path to save JSON file
    """
    # Handle both History object and dict
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    # Convert numpy arrays/values to native Python types
    clean_history = {}
    for key, values in history_dict.items():
        if isinstance(values, (list, np.ndarray)):
            clean_history[key] = [float(x) for x in values]
        else:
            clean_history[key] = float(values)
    
    # Add metadata
    metadata = {
        'training_history': clean_history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_epochs': len(clean_history.get('loss', [])),
        'best_val_accuracy': float(max(clean_history.get('val_accuracy', [0]))),
        'best_val_loss': float(min(clean_history.get('val_loss', [float('inf')])))
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"\n✓ Training history saved to {filepath}")
    except Exception as e:
        print(f"\n✗ Error saving training history: {e}")

def load_training_history(filepath=TRAINING_HISTORY_PATH):
    """
    Load training history from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: Training history
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both old and new format
        if 'training_history' in data:
            return data['training_history']
        return data
    except FileNotFoundError:
        print(f"✗ History file not found: {filepath}")
        return {}
    except Exception as e:
        print(f"✗ Error loading history: {e}")
        return {}

def plot_training_history(history, save_path=None, show_plot=True):
    """
    Plot training and validation metrics (accuracy, loss, learning rate)
    
    Args:
        history: Training history (dict or History object)
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
    """
    if hasattr(history, 'history'):
        history = history.history
    
    # Determine number of subplots based on available metrics
    num_plots = 2
    if 'lr' in history:
        num_plots = 3
    
    fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 5))
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot 1: Accuracy
    axes[0].plot(history['accuracy'], label='Training', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=4)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Add best accuracy annotation
    best_val_acc = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val_acc)
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    axes[0].text(best_epoch, best_val_acc, f' Best: {best_val_acc:.4f}', 
                fontsize=9, color='red')
    
    # Plot 2: Loss
    axes[1].plot(history['loss'], label='Training', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add best loss annotation
    best_val_loss = min(history['val_loss'])
    best_loss_epoch = history['val_loss'].index(best_val_loss)
    axes[1].axvline(x=best_loss_epoch, color='red', linestyle='--', alpha=0.5)
    axes[1].text(best_loss_epoch, best_val_loss, f' Best: {best_val_loss:.4f}', 
                fontsize=9, color='red')
    
    # Plot 3: Learning Rate (if available)
    if num_plots == 3 and 'lr' in history:
        axes[2].plot(history['lr'], linewidth=2, marker='o', markersize=4, color='green')
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training plot saved to {save_path}")
        except Exception as e:
            print(f"✗ Error saving plot: {e}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix(cm, normalize=False, save_path=None, show_plot=True):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        normalize: Normalize to percentages
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        title = 'Confusion Matrix - CIFAR-10 Classification (Normalized %)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix - CIFAR-10 Classification'
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        square=True
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"✗ Error saving confusion matrix: {e}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_sample_predictions(model, x_test, y_test, num_samples=12, save_path=None):
    """
    Plot sample predictions with true and predicted labels
    
    Args:
        model: Trained model
        x_test: Test images (preprocessed)
        y_test: Test labels (one-hot encoded)
        num_samples: Number of samples to display
        save_path: Path to save plot
    """
    # Get random samples
    indices = np.random.choice(len(x_test), min(num_samples, len(x_test)), replace=False)
    
    # Make predictions
    print("Making predictions on sample images...")
    predictions = model.predict(x_test[indices], verbose=0)
    
    # Create subplot grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Get image and labels
        img = x_test[idx]
        
        # Denormalize image if needed (for display)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
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
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Sample predictions saved to {save_path}")
        except Exception as e:
            print(f"✗ Error saving predictions: {e}")
    
    plt.show()

def plot_class_distribution(y_data, title='Class Distribution', save_path=None):
    """
    Plot class distribution bar chart
    
    Args:
        y_data: Labels (one-hot encoded or integer)
        title: Plot title
        save_path: Path to save plot
    """
    # Handle both one-hot and integer labels
    if len(y_data.shape) > 1 and y_data.shape[1] > 1:
        class_counts = np.sum(y_data, axis=0)
    else:
        class_counts = np.bincount(y_data.flatten(), minlength=NUM_CLASSES)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CLASS_NAMES, class_counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Class distribution plot saved to {save_path}")
        except Exception as e:
            print(f"✗ Error saving plot: {e}")
    
    plt.show()

def create_callbacks(model_save_path, monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE):
    """
    Create comprehensive training callbacks
    
    Args:
        model_save_path: Path to save best model
        monitor: Metric to monitor
        patience: Early stopping patience
    
    Returns:
        list: List of callbacks
    """
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor=monitor,
            save_best_only=True,
            mode='max' if 'accuracy' in monitor else 'min',
            verbose=1,
            save_weights_only=False
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
            cooldown=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, f'run_{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),
        
        # Custom callback for progress tracking
        ProgressCallback()
    ]
    
    return callbacks

class ProgressCallback(keras.callbacks.Callback):
    """Custom callback to track training progress and time"""
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        print(f"\n{'='*70}")
        print("TRAINING STARTED")
        print(f"{'='*70}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Estimate time remaining
        avg_time = np.mean(self.epoch_times)
        epochs_remaining = self.params['epochs'] - (epoch + 1)
        time_remaining = avg_time * epochs_remaining
        
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']} completed in {epoch_time:.1f}s")
        print(f"Estimated time remaining: {time_remaining/60:.1f} minutes")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Average time per epoch: {np.mean(self.epoch_times):.1f} seconds")
        print(f"{'='*70}\n")

def get_model_size(model_path):
    """
    Get model file size in MB
    
    Args:
        model_path: Path to model file
    
    Returns:
        float: Model size in MB
    """
    try:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        return 0.0
    except Exception as e:
        print(f"✗ Error getting model size: {e}")
        return 0.0

def print_evaluation_summary(results, save_path=None):
    """
    Print and optionally save formatted evaluation summary
    
    Args:
        results: Dictionary of evaluation results
        save_path: Path to save summary as text file
    """
    summary_lines = []
    summary_lines.append("\n" + "="*70)
    summary_lines.append("EVALUATION SUMMARY")
    summary_lines.append("="*70)
    
    for key, value in results.items():
        if isinstance(value, float):
            line = f"{key:<30}: {value:.4f}"
        elif isinstance(value, int):
            line = f"{key:<30}: {value:,}"
        else:
            line = f"{key:<30}: {value}"
        summary_lines.append(line)
    
    summary_lines.append("="*70)
    
    # Print to console
    for line in summary_lines:
        print(line)
    
    # Save to file if requested
    if save_path:
        try:
            with open(save_path, 'w') as f:
                f.write('\n'.join(summary_lines))
            print(f"\n✓ Evaluation summary saved to {save_path}")
        except Exception as e:
            print(f"\n✗ Error saving summary: {e}")

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels (one-hot or integer)
        y_pred: Predicted labels (one-hot or integer)
    
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Convert to integer labels if one-hot
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics

def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def cleanup_old_logs(log_dir=LOG_DIR, keep_last=5):
    """
    Clean up old TensorBoard logs, keeping only the most recent runs
    
    Args:
        log_dir: Directory containing logs
        keep_last: Number of recent runs to keep
    """
    try:
        if not os.path.exists(log_dir):
            return
        
        # Get all run directories
        runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        runs.sort()
        
        # Remove old runs
        if len(runs) > keep_last:
            for old_run in runs[:-keep_last]:
                old_path = os.path.join(log_dir, old_run)
                import shutil
                shutil.rmtree(old_path)
                print(f"✓ Removed old log: {old_run}")
    except Exception as e:
        print(f"✗ Error cleaning up logs: {e}")