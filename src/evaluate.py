"""
Evaluation script for CIFAR-10 classification model
Enhanced with proper dataset handling and comprehensive analysis
"""

import argparse
import os
import sys
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_loader import load_cifar10_data, create_test_dataset
from utils import (
    plot_confusion_matrix,
    plot_sample_predictions,
    print_evaluation_summary,
    get_model_size,
    calculate_metrics
)

def evaluate_model(model_path, detailed=False, save_results=True):
    """
    Comprehensive evaluation of trained model on test set
    
    Args:
        model_path: Path to saved model
        detailed: Whether to show detailed analysis with visualizations
        save_results: Whether to save results to files
    
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION - CIFAR-10 CLASSIFICATION")
    print("="*70)
    
    # Step 1: Load model
    print("\n" + "="*70)
    print("STEP 1: LOADING MODEL")
    print("="*70)
    
    try:
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Please ensure the model file exists at: {model_path}")
        return None
    
    model_size = get_model_size(model_path)
    print(f"Model size: {model_size:.2f} MB")
    print(f"Model name: {model.name}")
    
    # Print model summary
    print("\nModel architecture:")
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    
    # Step 2: Load and prepare test data
    print("\n" + "="*70)
    print("STEP 2: LOADING TEST DATA")
    print("="*70)
    
    try:
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
        print(f"✓ Test data loaded: {len(x_test)} samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Create test dataset with proper preprocessing
    print("\nCreating preprocessed test dataset...")
    test_dataset = create_test_dataset(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE
    )
    
    # Also keep raw test data for visualization
    y_test_categorical = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Step 3: Model evaluation
    print("\n" + "="*70)
    print("STEP 3: EVALUATING MODEL")
    print("="*70)
    
    print("Running evaluation on test dataset...")
    try:
        test_metrics = model.evaluate(
            test_dataset,
            verbose=1,
            return_dict=True
        )
        
        print(f"\n{'='*70}")
        print("MODEL METRICS ON TEST SET")
        print(f"{'='*70}")
        for metric_name, metric_value in test_metrics.items():
            if 'loss' in metric_name:
                print(f"{metric_name:<25}: {metric_value:.4f}")
            else:
                print(f"{metric_name:<25}: {metric_value:.4f} ({metric_value*100:.2f}%)")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        test_metrics = {}
    
    # Step 4: Generate predictions for detailed analysis
    print("\n" + "="*70)
    print("STEP 4: GENERATING PREDICTIONS")
    print("="*70)
    
    print("Making predictions on test set...")
    try:
        # Create a version of test dataset for prediction
        predict_dataset = create_test_dataset(
            x_test,
            y_test,
            batch_size=BATCH_SIZE,
            target_size=IMAGE_SIZE
        )
        
        predictions = model.predict(predict_dataset, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = y_test.flatten()
        
        print(f"✓ Predictions generated for {len(y_pred)} samples")
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None
    
    # Step 5: Calculate detailed metrics
    print("\n" + "="*70)
    print("STEP 5: CALCULATING METRICS")
    print("="*70)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0
    )
    
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")
    
    print("-" * 70)
    
    # Average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print(f"\n{'Macro Average':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    
    # Weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print(f"{'Weighted Average':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    
    # Step 6: Confusion Matrix
    print("\n" + "="*70)
    print("STEP 6: GENERATING CONFUSION MATRIX")
    print("="*70)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if save_results:
        # Save regular confusion matrix
        cm_save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        plot_confusion_matrix(cm, normalize=False, save_path=cm_save_path, show_plot=False)
        
        # Save normalized confusion matrix
        cm_norm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(cm, normalize=True, save_path=cm_norm_path, show_plot=False)
        
        print(f"✓ Confusion matrices saved to {RESULTS_DIR}/")
    
    # Step 7: Classification Report
    print("\n" + "="*70)
    print("STEP 7: GENERATING CLASSIFICATION REPORT")
    print("="*70)
    
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    
    print("\n" + report)
    
    if save_results:
        report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
        try:
            with open(report_path, 'w') as f:
                f.write("CIFAR-10 Classification Report\n")
                f.write("="*70 + "\n\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Model Size: {model_size:.2f} MB\n")
                f.write(f"Test Samples: {len(y_test)}\n\n")
                f.write(report)
            print(f"✓ Classification report saved to: {report_path}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
    
    # Step 8: Best and Worst Performing Classes
    print("\n" + "="*70)
    print("STEP 8: CLASS PERFORMANCE ANALYSIS")
    print("="*70)
    
    class_accuracies = []
    for i in range(NUM_CLASSES):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            class_accuracies.append((CLASS_NAMES[i], class_acc, np.sum(class_mask)))
    
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 BEST PERFORMING CLASSES")
    print("-" * 70)
    for i in range(min(3, len(class_accuracies))):
        name, acc, count = class_accuracies[i]
        print(f"{i+1}. {name:<15}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
    
    print("\n⚠️  WORST PERFORMING CLASSES")
    print("-" * 70)
    for i in range(min(3, len(class_accuracies))):
        name, acc, count = class_accuracies[-(i+1)]
        print(f"{i+1}. {name:<15}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
    
    # Step 9: Detailed Analysis (if requested)
    if detailed:
        print("\n" + "="*70)
        print("STEP 9: DETAILED ANALYSIS")
        print("="*70)
        
        # Prepare test data for visualization
        print("\nPreparing test data for visualization...")
        x_test_viz = []
        for i in range(min(12, len(x_test))):
            img = x_test[i]
            # Resize for visualization
            img_resized = tf.image.resize(img, IMAGE_SIZE).numpy()
            x_test_viz.append(img_resized)
        x_test_viz = np.array(x_test_viz) / 255.0  # Normalize
        y_test_viz = y_test_categorical[:len(x_test_viz)]
        
        # Sample predictions visualization
        if save_results:
            sample_path = os.path.join(RESULTS_DIR, 'sample_predictions.png')
            plot_sample_predictions(model, x_test_viz, y_test_viz, num_samples=12, save_path=sample_path)
        else:
            plot_sample_predictions(model, x_test_viz, y_test_viz, num_samples=12)
        
        # Most confident correct predictions
        print("\n✓ TOP 5 MOST CONFIDENT CORRECT PREDICTIONS")
        print("-" * 70)
        correct_mask = y_true == y_pred
        if np.sum(correct_mask) > 0:
            correct_confidences = np.max(predictions[correct_mask], axis=1)
            top_correct_indices = np.argsort(correct_confidences)[-5:][::-1]
            
            for rank, idx in enumerate(top_correct_indices, 1):
                actual_idx = np.where(correct_mask)[0][idx]
                conf = correct_confidences[idx]
                label = CLASS_NAMES[y_true[actual_idx]]
                print(f"{rank}. {label:<15}: {conf*100:.2f}% confidence")
        
        # Most confident incorrect predictions
        print("\n❌ TOP 5 MOST CONFIDENT INCORRECT PREDICTIONS")
        print("-" * 70)
        incorrect_mask = y_true != y_pred
        if np.sum(incorrect_mask) > 0:
            incorrect_confidences = np.max(predictions[incorrect_mask], axis=1)
            top_incorrect_indices = np.argsort(incorrect_confidences)[-5:][::-1]
            
            for rank, idx in enumerate(top_incorrect_indices, 1):
                actual_idx = np.where(incorrect_mask)[0][idx]
                conf = incorrect_confidences[idx]
                true_label = CLASS_NAMES[y_true[actual_idx]]
                pred_label = CLASS_NAMES[y_pred[actual_idx]]
                print(f"{rank}. Predicted: {pred_label:<10} | True: {true_label:<10} | Confidence: {conf*100:.2f}%")
        
        # Confusion pairs analysis
        print("\n🔀 MOST CONFUSED CLASS PAIRS")
        print("-" * 70)
        confused_pairs = []
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        for rank, (true_class, pred_class, count) in enumerate(confused_pairs[:5], 1):
            print(f"{rank}. {true_class:<10} → {pred_class:<10}: {count} times")
    
    # Step 10: Generate Summary
    print("\n" + "="*70)
    print("STEP 10: FINAL SUMMARY")
    print("="*70)
    
    results = {
        'Model Path': model_path,
        'Model Name': model.name,
        'Model Size (MB)': f"{model_size:.2f}",
        'Total Parameters': f"{total_params:,}",
        'Test Samples': len(x_test),
        'Overall Accuracy': accuracy,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average F1-Score': avg_f1,
        'Weighted Precision': weighted_precision,
        'Weighted Recall': weighted_recall,
        'Weighted F1-Score': weighted_f1,
        'Best Class': class_accuracies[0][0],
        'Best Class Accuracy': class_accuracies[0][1],
        'Worst Class': class_accuracies[-1][0],
        'Worst Class Accuracy': class_accuracies[-1][1]
    }
    
    # Add test metrics if available
    if test_metrics:
        for key, value in test_metrics.items():
            results[f'Test {key}'] = value
    
    if save_results:
        summary_path = os.path.join(RESULTS_DIR, 'evaluation_summary.txt')
        print_evaluation_summary(results, save_path=summary_path)
    else:
        print_evaluation_summary(results)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    if save_results:
        print(f"\n📁 Results saved to: {RESULTS_DIR}/")
        print("   • confusion_matrix.png")
        print("   • confusion_matrix_normalized.png")
        print("   • classification_report.txt")
        print("   • evaluation_summary.txt")
        if detailed:
            print("   • sample_predictions.png")
    
    return results

def main():
    """
    Main function to parse arguments and evaluate model
    """
    parser = argparse.ArgumentParser(
        description='Evaluate CIFAR-10 Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=MOBILENETV2_MODEL_PATH,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed analysis with visualizations'
    )
    
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results to files'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("\nAvailable models:")
        if os.path.exists(MODELS_DIR):
            models = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.h5', '.keras'))]
            if models:
                for model in models:
                    print(f"  • {os.path.join(MODELS_DIR, model)}")
            else:
                print("  No trained models found")
        sys.exit(1)
    
    # Evaluate model
    try:
        results = evaluate_model(
            args.model_path,
            detailed=args.detailed,
            save_results=not args.no_save
        )
        
        if results is None:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()