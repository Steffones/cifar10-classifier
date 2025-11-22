#!/bin/bash

echo "=========================================="
echo "CIFAR-10 Classifier Project Setup"
echo "=========================================="

# Check if my_ml_env exists
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Error: No Python virtual environment activated!"
    echo "Please activate my_ml_env first:"
    echo ""
    echo "   source ~/my_ml_env/bin/activate"
    echo ""
    exit 1
fi

# Activate environment
echo "✓ Activating my_ml_env..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source my_ml_env/Scripts/activate
else
    # macOS/Linux
    source my_ml_env/bin/activate
fi

# Create directory structure
echo "✓ Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p results/plots
mkdir -p notebooks

# Create __init__.py files
touch src/__init__.py
touch app/__init__.py

# Install dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo ""
echo "✓ Verifying installations..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Train model: bash train.sh"
echo "2. Or run: python src/train.py"
echo ""