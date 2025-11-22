"""
Evaluation script for CIFAR-10 classification model
"""

import argparse
import numpy as np
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from config import *
from data_loader import load_cifar10_data, preprocess_data
from utils import (
    plot_confusion_matrix,
    plot_sample_predictions,
    print_evaluation_summary,
    get_model_size
)

def evaluate_model(model_path, detailed=False):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to saved model
        detailed: Whether to show detailed analysis
    
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION - CIFAR-10 CLASSIFICATION")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    model_size = get_model_size(model_path)
    print(f"Model size: {model_size:.2f} MB")
    
    # Load and preprocess data
    print("\nLoading test data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    _, _, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = model.predict(x_test, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(NUM_CLASSES)
    )
    
    print("\nPer-Class Performance:")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-" * 70)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]}")
    
    print("-" * 70)
    
    # Average metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print(f"\n{'Average':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    cm_save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, save_path=cm_save_path)
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save classification report
    report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Classification report saved to: {report_path}")
    
    # Find best and worst performing classes
    class_accuracies = []
    for i in range(NUM_CLASSES):
        class_mask = y_true == i
        class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
        class_accuracies.append((CLASS_NAMES[i], class_acc))
    
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*70)
    print("BEST PERFORMING CLASSES")
    print("="*70)
    for i in range(min(3, len(class_accuracies))):
        name, acc = class_accuracies[i]
        print(f"{i+1}. {name}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n" + "="*70)
    print("WORST PERFORMING CLASSES")
    print("="*70)
    for i in range(min(3, len(class_accuracies))):
        name, acc = class_accuracies[-(i+1)]
        print(f"{i+1}. {name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Detailed analysis
    if detailed:
        print("\n" + "="*70)
        print("DETAILED ANALYSIS")
        print("="*70)
        
        # Sample predictions
        print("\nSample predictions (random selection):")
        plot_sample_predictions(model, x_test, y_test, num_samples=12)
        
        # Most confident correct predictions
        correct_mask = y_true == y_pred
        correct_confidences = np.max(predictions[correct_mask], axis=1)
        top_correct_indices = np.argsort(correct_confidences)[-5:][::-1]
        
        print("\nTop 5 Most Confident Correct Predictions:")
        for idx in top_correct_indices:
            actual_idx = np.where(correct_mask)[0][idx]
            conf = correct_confidences[idx]
            label = CLASS_NAMES[y_true[actual_idx]]
            print(f"  - {label}: {conf*100:.2f}% confidence")
        
        # Most confident incorrect predictions
        incorrect_mask = y_true != y_pred
        if np.any(incorrect_mask):
            incorrect_confidences = np.max(predictions[incorrect_mask], axis=1)
            top_incorrect_indices = np.argsort(incorrect_confidences)[-5:][::-1]
            
            print("\nTop 5 Most Confident Incorrect Predictions:")
            for idx in top_incorrect_indices:
                actual_idx = np.where(incorrect_mask)[0][idx]
                conf = incorrect_confidences[idx]
                true_label = CLASS_NAMES[y_true[actual_idx]]
                pred_label = CLASS_NAMES[y_pred[actual_idx]]
                print(f"  - Predicted {pred_label} (was {true_label}): {conf*100:.2f}% confidence")
    
    # Summary
    results = {
        'Model Path': model_path,
        'Model Size (MB)': f"{model_size:.2f}",
        'Test Samples': len(x_test),
        'Overall Accuracy': accuracy,
        'Average Precision': avg_precision,
        'Average Recall': avg_recall,
        'Average F1-Score': avg_f1,
        'Best Class': class_accuracies[0][0],
        'Best Class Accuracy': class_accuracies[0][1],
        'Worst Class': class_accuracies[-1][0],
        'Worst Class Accuracy': class_accuracies[-1][1]
    }
    
    print_evaluation_summary(results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE! ✅")
    print("="*70)
    
    return results

def main():
    """
    Main function to parse arguments and evaluate model
    """
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 Classification Model')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=MOBILENETV2_MODEL_PATH,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed analysis'
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(args.model_path, detailed=args.detailed)

if __name__ == '__main__':
    main()