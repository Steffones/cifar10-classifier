"""
Training script for CIFAR-10 classification
Optimized for memory efficiency and robust training
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *

# Configure TensorFlow for optimal performance
if ALLOW_GROWTH:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(physical_devices)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration warning: {e}")

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from data_loader import (
    load_cifar10_data,
    create_data_generators,
    create_test_dataset
)
from model_builder import (
    build_mobilenetv2_model,
    build_custom_cnn_model,
    compile_model,
    unfreeze_model_layers,
    print_model_summary,
    get_model_config
)
from utils import (
    create_callbacks,
    plot_training_history,
    save_training_history,
    get_model_size,
    format_time
)

def print_system_info():
    """Print system and TensorFlow configuration"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    #print(f"Keras version: {keras.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("GPUs available: 0 (CPU-only)")
    
    # Memory info
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Total RAM: {ram_gb:.1f} GB")
    print(f"Available RAM: {available_gb:.1f} GB")
    print("="*70)

def train_model(model_type='mobilenetv2', epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Complete training pipeline with two-phase training for transfer learning
    
    Args:
        model_type: Type of model ('mobilenetv2' or 'custom_cnn')
        epochs: Total number of training epochs
        batch_size: Batch size for training
    
    Returns:
        tuple: (trained_model, training_history)
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*70)
    print(f"CIFAR-10 CLASSIFICATION TRAINING")
    print(f"Model: {model_type.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Print system info
    print_system_info()
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    
    # Step 2: Create datasets using tf.data API
    print("\n" + "="*70)
    print("STEP 2: CREATING DATA PIPELINES")
    print("="*70)
    
    train_dataset, val_dataset, steps_per_epoch, validation_steps = create_data_generators(
        x_train, y_train,
        batch_size=batch_size,
        target_size=IMAGE_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Create test dataset
    test_dataset = create_test_dataset(
        x_test, y_test,
        batch_size=batch_size,
        target_size=IMAGE_SIZE
    )
    
    # Step 3: Build model
    print("\n" + "="*70)
    print("STEP 3: BUILDING MODEL")
    print("="*70)
    
    if model_type.lower() == 'mobilenetv2':
        model = build_mobilenetv2_model(
            input_shape=IMAGE_SIZE + (3,),
            trainable_base=BASE_MODEL_TRAINABLE_PHASE1
        )
        model_save_path = MOBILENETV2_MODEL_PATH
        use_fine_tuning = True
    elif model_type.lower() == 'custom_cnn':
        model = build_custom_cnn_model(input_shape=IMAGE_SIZE + (3,))
        model_save_path = CUSTOM_CNN_MODEL_PATH
        use_fine_tuning = False
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'mobilenetv2' or 'custom_cnn'")
    
    # Step 4: Compile model
    print("\n" + "="*70)
    print("STEP 4: COMPILING MODEL")
    print("="*70)
    
    model = compile_model(model, learning_rate=LEARNING_RATE)
    print_model_summary(model, show_trainable=True)
    
    # Save model config
    model_config = get_model_config(model)
    print(f"\nModel configuration saved")
    
    # Step 5: Create callbacks
    print("\n" + "="*70)
    print("STEP 5: PREPARING TRAINING CALLBACKS")
    print("="*70)
    
    callbacks = create_callbacks(
        model_save_path=model_save_path,
        monitor='val_accuracy',
        patience=EARLY_STOPPING_PATIENCE
    )
    
    print(f"Callbacks configured:")
    print(f"  • Model checkpoint (save best)")
    print(f"  • Early stopping (patience={EARLY_STOPPING_PATIENCE})")
    print(f"  • Learning rate reduction")
    print(f"  • TensorBoard logging")
    print(f"  • Progress tracking")
    
    # Calculate training phases
    if use_fine_tuning:
        phase1_epochs = PHASE_1_EPOCHS
        phase2_epochs = PHASE_2_EPOCHS
        total_epochs = phase1_epochs + phase2_epochs
    else:
        phase1_epochs = epochs
        phase2_epochs = 0
        total_epochs = epochs
    
    print(f"\nTraining plan:")
    print(f"  Phase 1 (Feature extraction): {phase1_epochs} epochs")
    if use_fine_tuning:
        print(f"  Phase 2 (Fine-tuning): {phase2_epochs} epochs")
    print(f"  Total: {total_epochs} epochs")
    
    # Step 6: Training Phase 1 - Feature Extraction
    print("\n" + "="*70)
    print("STEP 6: PHASE 1 - FEATURE EXTRACTION")
    print("="*70)
    print(f"Training with frozen base model...")
    print(f"Epochs: {phase1_epochs}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print("="*70)
    
    phase1_start = time.time()
    
    try:
        history_phase1 = model.fit(
            train_dataset,
            epochs=phase1_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=VERBOSE
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        return None, None
    except Exception as e:
        print(f"\n❌ Error during Phase 1 training: {e}")
        raise
    
    phase1_time = time.time() - phase1_start
    print(f"\n✓ Phase 1 completed in {format_time(phase1_time)}")
    
    # Step 7: Training Phase 2 - Fine Tuning (if applicable)
    history_phase2 = None
    
    if use_fine_tuning:
        print("\n" + "="*70)
        print("STEP 7: PHASE 2 - FINE-TUNING")
        print("="*70)
        
        # Unfreeze layers
        model = unfreeze_model_layers(model, num_layers_to_unfreeze=UNFREEZE_LAYERS)
        
        # Recompile with lower learning rate
        model = compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
        
        print(f"\nFine-tuning configuration:")
        print(f"  Epochs: {phase2_epochs}")
        print(f"  Learning rate: {FINE_TUNE_LEARNING_RATE}")
        print(f"  Unfrozen layers: {UNFREEZE_LAYERS}")
        print("="*70)
        
        phase2_start = time.time()
        
        try:
            history_phase2 = model.fit(
                train_dataset,
                epochs=phase1_epochs + phase2_epochs,
                initial_epoch=phase1_epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=VERBOSE
            )
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user during Phase 2!")
        except Exception as e:
            print(f"\n❌ Error during Phase 2 training: {e}")
            print("Continuing with Phase 1 model...")
        
        phase2_time = time.time() - phase2_start
        print(f"\n✓ Phase 2 completed in {format_time(phase2_time)}")
        
        # Combine histories
        if history_phase2 is not None:
            print("\nCombining training histories...")
            combined_history = {}
            for key in history_phase1.history.keys():
                combined_history[key] = (
                    history_phase1.history[key] + 
                    history_phase2.history[key]
                )
            
            # Create combined history object
            class CombinedHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            history = CombinedHistory(combined_history)
        else:
            history = history_phase1
    else:
        history = history_phase1
    
    # Step 8: Load best model
    print("\n" + "="*70)
    print("STEP 8: LOADING BEST MODEL")
    print("="*70)
    
    try:
        print(f"Loading best model from: {model_save_path}")
        model = keras.models.load_model(model_save_path)
        print("✓ Best model loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load best model: {e}")
        print("Using current model state instead")
    
    # Step 9: Final evaluation on test set
    print("\n" + "="*70)
    print("STEP 9: FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    try:
        print("Evaluating on test dataset...")
        test_results = model.evaluate(
            test_dataset,
            verbose=1,
            return_dict=True
        )
        
        print(f"\n{'='*70}")
        print("TEST SET RESULTS")
        print(f"{'='*70}")
        for metric_name, metric_value in test_results.items():
            if 'loss' in metric_name:
                print(f"{metric_name:<30}: {metric_value:.4f}")
            else:
                print(f"{metric_name:<30}: {metric_value:.4f} ({metric_value*100:.2f}%)")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"⚠️ Error during test evaluation: {e}")
        print("Skipping test evaluation")
        test_results = {}
    
    # Step 10: Save training history
    print("\n" + "="*70)
    print("STEP 10: SAVING TRAINING ARTIFACTS")
    print("="*70)
    
    save_training_history(history)
    
    # Step 11: Plot training curves
    try:
        plot_path = os.path.join(PLOTS_DIR, f'{model_type}_training_history.png')
        plot_training_history(history, save_path=plot_path, show_plot=False)
    except Exception as e:
        print(f"⚠️ Could not save training plot: {e}")
    
    # Step 12: Print final summary
    total_time = time.time() - start_time
    model_size = get_model_size(model_save_path)
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model type: {model_type}")
    print(f"Total training time: {format_time(total_time)}")
    if use_fine_tuning:
        print(f"  Phase 1 time: {format_time(phase1_time)}")
        print(f"  Phase 2 time: {format_time(phase2_time)}")
    print(f"Model saved to: {model_save_path}")
    print(f"Model size: {model_size:.2f} MB")
    
    if test_results:
        print(f"\nFinal test accuracy: {test_results.get('accuracy', 0):.4f} ({test_results.get('accuracy', 0)*100:.2f}%)")
        if 'top_3_accuracy' in test_results:
            print(f"Final test top-3 accuracy: {test_results.get('top_3_accuracy', 0):.4f} ({test_results.get('top_3_accuracy', 0)*100:.2f}%)")
    
    print("="*70)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📝 Next steps:")
    print(f"   1. Evaluate model:  python src/evaluate.py --model_path {model_save_path}")
    print(f"   2. Run Streamlit:   streamlit run app/streamlit_app.py")
    print(f"   3. Run Gradio:      python app/gradio_app.py")
    print(f"   4. View logs:       tensorboard --logdir {LOG_DIR}")
    
    return model, history

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train CIFAR-10 Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
        help='Total number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--no_fine_tune',
        action='store_true',
        help='Skip fine-tuning phase (Phase 2)'
    )
    
    args = parser.parse_args()
    
    # Override epochs if specified
    if args.epochs != EPOCHS:
        print(f"Using custom epochs: {args.epochs}")
    
    try:
        model, history = train_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if model is None:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()