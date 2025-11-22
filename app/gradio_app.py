"""
Streamlit app for CIFAR-10 classification
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

from src.config import *
from src.data_loader import preprocess_single_image

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load trained model (cached)"""
    try:
        model = keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def predict_image(model, image):
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        image: PIL Image
    
    Returns:
        tuple: (predicted_class, confidence, all_predictions)
    """
    # Preprocess image
    img_array = preprocess_single_image(image)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    return predicted_class, confidence, predictions

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ CIFAR-10 Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to classify it into one of 10 categories</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_options = {
        "MobileNetV2 (Recommended)": MOBILENETV2_MODEL_PATH,
        "Custom CNN": CUSTOM_CNN_MODEL_PATH
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    model_path = model_options[selected_model_name]
    
    # Load model
    with st.spinner("Loading model..."):
        model, error = load_model(model_path)
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Please train the model first by running: `python src/train.py`")
        return
    
    st.sidebar.success(f"✅ Model loaded: {selected_model_name}")
    
    # Show model info
    if st.sidebar.checkbox("Show Model Info"):
        st.sidebar.write(f"**Model Architecture:** {selected_model_name}")
        st.sidebar.write(f"**Input Size:** {IMAGE_SIZE}")
        st.sidebar.write(f"**Number of Classes:** {NUM_CLASSES}")
    
    # Class names reference
    with st.sidebar.expander("📋 Available Classes"):
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. {class_name.capitalize()}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to classify"
        )
        
        # Sample images
        st.subheader("Or try a sample image:")
        sample_cols = st.columns(5)
        
        # You can add sample images here if you have them
        sample_selected = None
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, all_predictions = predict_image(model, image)
                
                # Store in session state
                st.session_state['prediction'] = {
                    'class': predicted_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions,
                    'image': image
                }
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            
            # Main prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #1f77b4; margin: 0;">Predicted Class</h2>
                    <h1 style="color: #2ecc71; margin: 10px 0;">{pred['class'].upper()}</h1>
                    <h3 style="color: #555; margin: 0;">Confidence: {pred['confidence']:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            st.subheader("Confidence Level")
            st.progress(pred['confidence'] / 100)
            
            # Top 5 predictions
            st.subheader("Top 5 Predictions")
            
            # Create dataframe for top predictions
            top_indices = np.argsort(pred['all_predictions'])[-5:][::-1]
            top_classes = [CLASS_NAMES[i] for i in top_indices]
            top_confidences = [pred['all_predictions'][i] * 100 for i in top_indices]
            
            df = pd.DataFrame({
                'Class': top_classes,
                'Confidence (%)': top_confidences
            })
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Bar chart of all predictions
            st.subheader("All Class Probabilities")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(CLASS_NAMES, pred['all_predictions'] * 100, color='steelblue')
            
            # Highlight predicted class
            predicted_idx = CLASS_NAMES.index(pred['class'])
            bars[predicted_idx].set_color('#2ecc71')
            
            ax.set_xlabel('Confidence (%)', fontsize=12)
            ax.set_ylabel('Class', fontsize=12)
            ax.set_title('Prediction Probabilities for All Classes', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            
        else:
            st.info("👆 Upload an image and click 'Classify Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p><strong>CIFAR-10 Object Classification</strong></p>
            <p>Built with TensorFlow/Keras and Streamlit</p>
            <p>Transfer Learning with MobileNetV2</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()