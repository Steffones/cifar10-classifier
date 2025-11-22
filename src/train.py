"""
Training script for CIFAR-10 classification
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from config import *
from data_loader import load_cifar10_data, preprocess_data, create_data_generators
from model_builder import (
    build_mobilenetv2_model,
    build_custom_cnn_model,
    compile_model,
    unfreeze_model_layers,
    print_model_summary
)
from utils import (
    create_callbacks,
    plot_training_history,
    save_training_history,
    get_model_size
)

def train_model(model_type='mobilenetv2', epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Complete training pipeline
    
    Args:
        model_type: Type of model ('mobilenetv2' or 'custom_cnn')
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        tuple: (model, history)
    """
    print("\n" + "="*70)
    print(f"CIFAR-10 CLASSIFICATION TRAINING - {model_type.upper()}")
    print("="*70)
    
    # Step 1: Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    
    # Step 2: Preprocess data
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # Step 3: Create data generators
    train_gen, val_gen = create_data_generators(x_train, y_train, batch_size)
    
    # Step 4: Build model
    if model_type.lower() == 'mobilenetv2':
        model = build_mobilenetv2_model()
        model_save_path = MOBILENETV2_MODEL_PATH
    elif model_type.lower() == 'custom_cnn':
        model = build_custom_cnn_model()
        model_save_path = CUSTOM_CNN_MODEL_PATH
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Step 5: Compile model
    model = compile_model(model, learning_rate=LEARNING_RATE)
    print_model_summary(model)
    
    # Step 6: Create callbacks
    callbacks = create_callbacks(model_save_path)
    
    # Step 7: Training Phase 1 - Feature Extraction
    print("\n" + "="*70)
    print("PHASE 1: FEATURE EXTRACTION (Frozen Base Model)")
    print("="*70)
    
    phase1_epochs = epochs // 2
    
    history_phase1 = model.fit(
        train_gen,
        epochs=phase1_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 8: Training Phase 2 - Fine Tuning (for MobileNetV2)
    if model_type.lower() == 'mobilenetv2':
        print("\n" + "="*70)
        print("PHASE 2: FINE-TUNING (Unfrozen Top Layers)")
        print("="*70)
        
        # Unfreeze layers
        model = unfreeze_model_layers(model, num_layers_to_unfreeze=30)
        
        # Recompile with lower learning rate
        model = compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
        
        # Continue training
        history_phase2 = model.fit(
            train_gen,
            epochs=epochs - phase1_epochs,
            initial_epoch=phase1_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        combined_history = {}
        for key in history_phase1.history.keys():
            combined_history[key] = (
                history_phase1.history[key] + history_phase2.history[key]
            )
        
        # Create a mock history object
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        history = CombinedHistory(combined_history)
    else:
        history = history_phase1
    
    # Step 9: Load best model
    print("\nLoading best model...")
    model = keras.models.load_model(model_save_path)
    
    # Step 10: Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(
        x_test, y_test, verbose=1
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)")
    
    # Step 11: Save training history
    save_training_history(history)
    
    # Step 12: Plot training curves
    plot_path = os.path.join(PLOTS_DIR, f'{model_type}_training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Step 13: Print model info
    model_size = get_model_size(model_save_path)
    print(f"\n✓ Model saved to: {model_save_path}")
    print(f"✓ Model size: {model_size:.2f} MB")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE! 🎉")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Evaluate: python src/evaluate.py --model_path {model_save_path}")
    print(f"2. Deploy: streamlit run app/streamlit_app.py")
    
    return model, history

def main():
    """
    Main function to parse arguments and start training
    """
    parser = argparse.ArgumentParser(description='Train CIFAR-10 Classification Model')
    
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenetv2',
        choices=['mobilenetv2', 'custom_cnn'],
        help='Model architecture to train'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for training'
    )
    
    args = parser.parse_args()
    
    # Start training
    model, history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()