#!/bin/bash

echo "=========================================="
echo "CIFAR-10 Model Training"
echo "=========================================="

# Ensure virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Error: No Python virtual environment activated!"
    echo "Please activate your ML environment first."
    exit 1
fi

echo "✓ Using environment: $VIRTUAL_ENV"

# Default parameters
MODEL="mobilenetv2"
EPOCHS=20
BATCH_SIZE=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  • Model: $MODEL"
echo "  • Epochs: $EPOCHS"
echo "  • Batch Size: $BATCH_SIZE"
echo ""

# Start training
python src/train.py --model "$MODEL" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Evaluate: python src/evaluate.py"
echo "2. Run app: streamlit run app/streamlit_app.py"
echo ""
