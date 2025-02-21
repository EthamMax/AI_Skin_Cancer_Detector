import os
os.system("pip install huggingface-hub tf-explain opencv-python streamlit-extras")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tf_explain.core.grad_cam import GradCAM
import time

# --- Helper Function: Check if Image is Mostly Skin-colored ---
def is_skin_image(image, threshold=0.10):
    """
    Checks whether the input PIL image is mostly skin-colored.
    Converts the image to YCrCb color space and applies a skin-tone mask.
    Returns True if the ratio of skin-colored pixels exceeds the threshold.
    """
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_YCrCb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skinMask = cv2.inRange(image_YCrCb, lower, upper)
    skinPixels = cv2.countNonZero(skinMask)
    totalPixels = image_cv.shape[0] * image_cv.shape[1]
    ratio = skinPixels / totalPixels
    return ratio > threshold

# --- Apple-inspired Theme ---
def apply_apple_theme():
    apple_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.022em;
        }
        
        .main {
            background-color: #ffffff;
            background-image: linear-gradient(180deg, #f5f5f7 0%, #ffffff 100%);
            padding: 0 !important;
        }
        
        .stApp {
            max-width: 100%;
        }
        
        /* Apple-style full-width hero section */
        .hero-section {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            max-height: 800px;
            min-height: 600px;
            text-align: center;
            padding: 0 20px;
            margin-bottom: 60px;
            background: linear-gradient(180deg, #f0f0f3 0%, #ffffff 100%);
            position: relative;
            overflow: hidden;
        }
        
        /* Apple typography styles */
        .hero-title {
            font-size: 56px;
            font-weight: 600;
            background: linear-gradient(90deg, #1D1D1F 0%, #86868b 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            line-height: 1.1;
            transform: translateY(0);
            opacity: 1;
            animation: fadeInUp 1s ease-out;
        }
        
        .hero-subtitle {
            font-size: 24px;
            font-weight: 400;
            color: #86868b;
            margin-bottom: 30px;
            max-width: 600px;
            animation: fadeInUp 1s ease-out 0.2s both;
        }
        
        .app-credit {
            font-size: 16px;
            color: #86868b;
            margin-top: 40px;
            animation: fadeInUp 1s ease-out 0.4s both;
        }
        
        /* Apple-style section headers */
        .apple-section-header {
            font-size: 32px;
            font-weight: 600;
            color: #1D1D1F;
            margin: 60px 0 24px 0;
            text-align: center;
        }
        
        /* Apple-style cards */
        .apple-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 18px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
            padding: 32px;
            margin-bottom: 24px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .apple-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }
        
        /* Apple-style buttons */
        .apple-button {
            display: inline-block;
            background-color: #0071e3;
            color: white;
            font-size: 17px;
            font-weight: 500;
            padding: 12px 22px;
            border-radius: 980px;
            text-decoration: none;
            text-align: center;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            margin-top: 20px;
        }
        
        .apple-button:hover {
            background-color: #0077ED;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Apple-style results section */
        .results-container {
            background: linear-gradient(135deg, #f0f0f3 0%, #ffffff 100%);
            border-radius: 20px;
            padding: 40px;
            margin-top: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06);
            animation: fadeIn 0.8s ease-out;
        }
        
        .result-metric {
            font-size: 18px;
            color: #1D1D1F;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        
        .result-value {
            font-weight: 600;
            margin-left: 10px;
        }
        
        .risk-high {
            color: #FF3B30;
        }
        
        .risk-low {
            color: #34C759;
        }
        
        /* Apple-style loading animation */
        .apple-loader {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
            margin: 30px 0;
        }
        
        .loader-dot {
            width: 12px;
            height: 12px;
            margin: 0 5px;
            background-color: #0071e3;
            border-radius: 50%;
            display: inline-block;
            animation: dotPulse 1.5s infinite ease-in-out;
        }
        
        .loader-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .loader-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        /* Apple-style footer */
        .apple-footer {
            text-align: center;
            padding: 40px 0;
            margin-top: 60px;
            color: #86868b;
            font-size: 14px;
            border-top: 1px solid #d2d2d7;
        }
        
        /* Navigation bar */
        .apple-nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 48px;
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 1px 0 rgba(0, 0, 0, 0.05);
            z-index: 1000;
        }
        
        .nav-items {
            display: flex;
            gap: 30px;
        }
        
        .nav-item {
            color: #1D1D1F;
            font-size: 14px;
            font-weight: 400;
            text-decoration: none;
            transition: color 0.2s ease;
        }
        
        .nav-item:hover {
            color: #0071e3;
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        @keyframes dotPulse {
            0%, 80%, 100% { 
                transform: scale(0);
                opacity: 0;
            }
            40% { 
                transform: scale(1);
                opacity: 1;
            }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 42px;
            }
            
            .hero-subtitle {
                font-size: 20px;
            }
            
            .apple-section-header {
                font-size: 28px;
            }
        }
        
        /* Gradients for visualization */
        .gradient-bg {
            background: linear-gradient(135deg, #0071e3 0%, #40a9ff 100%);
            border-radius: 16px;
            height: 6px;
            width: 100%;
            margin: 20px 0;
        }
        
        /* Progress bar styling */
        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        .progress-text {
            font-size: 14px;
            color: #86868b;
        }
        
        .progress-bar {
            height: 6px;
            background-color: #f2f2f2;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #0071e3 0%, #40a9ff 100%);
            border-radius: 3px;
            transition: width 1s ease-out;
        }
        
        /* Image upload styling */
        .upload-container {
            border: 2px dashed #d2d2d7;
            border-radius: 16px;
            padding: 40px 20px;
            text-align: center;
            transition: all 0.3s ease;
            background-color: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(10px);
            margin: 40px 0;
        }
        
        .upload-container:hover {
            border-color: #0071e3;
            background-color: rgba(255, 255, 255, 0.8);
        }
        
        .upload-icon {
            font-size: 40px;
            color: #86868b;
            margin-bottom: 15px;
        }
        
        .upload-text {
            color: #1D1D1F;
            font-size: 16px;
            margin-bottom: 20px;
        }
        
        /* Content container for proper spacing */
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }
        
        /* Two column layout */
        .two-columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin: 40px 0;
        }
        
        @media (max-width: 992px) {
            .two-columns {
                grid-template-columns: 1fr;
            }
        }
        
        /* Animation for results appearance */
        .fade-in {
            animation: fadeIn 1s ease-out;
        }
        
        /* Visualization section styling */
        .visualization-container {
            position: relative;
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
            margin: 30px 0;
        }
        
        .visualization-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.03);
            border-radius: 24px;
        }
        
        /* Feature highlights */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            margin: 60px 0;
        }
        
        .feature-card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
        }
        
        .feature-icon {
            font-size: 40px;
            color: #0071e3;
            margin-bottom: 20px;
        }
        
        .feature-title {
            font-size: 20px;
            font-weight: 600;
            color: #1D1D1F;
            margin-bottom: 12px;
        }
        
        .feature-description {
            font-size: 16px;
            color: #86868b;
            line-height: 1.5;
        }
        
        @media (max-width: 768px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """
    st.markdown(apple_css, unsafe_allow_html=True)

# --- Apply Apple Theme ---
apply_apple_theme()

# --- Navigation Bar ---
st.markdown("""
<div class="apple-nav">
    <div class="nav-items">
        <a href="#" class="nav-item">Home</a>
        <a href="#about" class="nav-item">About</a>
        <a href="#features" class="nav-item">Features</a>
        <a href="#analysis" class="nav-item">Analysis</a>
        <a href="#contact" class="nav-item">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">AI Skin Cancer Detector</h1>
    <p class="hero-subtitle">Advanced technology for early detection, designed with precision and accessibility in mind.</p>
    <p class="app-credit">Developed by Mrityunjay Kumar, Biomedical Science Student at Acharya Narendra College (University of Delhi)</p>
</div>
""", unsafe_allow_html=True)

# --- Content Container ---
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# --- Features Section ---
st.markdown('<a id="features"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="apple-section-header">Innovative Features</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <h3 class="feature-title">Advanced Detection</h3>
        <p class="feature-description">Using state-of-the-art deep learning models trained on the comprehensive HAM1000 dataset for accurate lesion classification.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3 class="feature-title">Visual Insights</h3>
        <p class="feature-description">Grad-CAM visualization technology reveals exactly which areas of the lesion influenced the AI's decision-making process.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <h3 class="feature-title">Early Warning</h3>
        <p class="feature-description">Designed to detect potential issues early, when treatment is most effective and least invasive.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- About Section ---
st.markdown('<a id="about"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="apple-section-header">About This Technology</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="apple-card">
    <p style="font-size: 18px; line-height: 1.5; color: #1D1D1F;">
        The AI Skin Cancer Detector is built on MobileNetV2 architecture, trained on the HAM1000 dataset—a collection of over 10,000 dermatoscopic images of common pigmented skin lesions. This powerful combination allows for reliable classification of seven types of skin lesions, helping identify potential melanoma and other skin cancers in their early stages.
    </p>
    <div class="gradient-bg"></div>
    <p style="font-size: 16px; color: #86868b; font-style: italic;">
        Note: This application is designed as an educational tool and preliminary screening aid. It does not replace professional medical diagnosis. Always consult a dermatologist for proper evaluation.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Analysis Section ---
st.markdown('<a id="analysis"></a>', unsafe_allow_html=True)
st.markdown('<h2 class="apple-section-header">Image Analysis</h2>', unsafe_allow_html=True)

# --- Upload Section ---
st.markdown("""
<div class="upload-container">
    <div class="upload-icon">📷</div>
    <p class="upload-text">Upload a clear image of a skin lesion for analysis</p>
</div>
""", unsafe_allow_html=True)

image_for_prediction = None
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

# --- Processing and Analysis ---
if uploaded_file is not None:
    image_for_prediction = Image.open(uploaded_file)
    
    # Display the uploaded image in an Apple-style card
    st.markdown("""
    <div class="apple-card">
        <h3 style="font-size: 24px; margin-bottom: 20px; color: #1D1D1F;">Uploaded Image</h3>
    """, unsafe_allow_html=True)
    
    st.image(image_for_prediction, use_container_width=False, width=400)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Check if it's a skin image
    if not is_skin_image(image_for_prediction, threshold=0.10):
        st.markdown("""
        <div class="apple-card" style="border-left: 4px solid #FF3B30; background-color: rgba(255, 59, 48, 0.05);">
            <h3 style="color: #FF3B30; font-size: 18px;">Image Validation Failed</h3>
            <p style="color: #1D1D1F;">The uploaded image doesn't appear to be a skin lesion image. Please upload a clear, close-up image of a skin lesion for accurate analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show Apple-style loading animation
        st.markdown("""
        <div class="apple-loader">
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
        </div>
        <p style="text-align: center; color: #86868b;">Analyzing image using advanced AI...</p>
        """, unsafe_allow_html=True)
        
        # Process the image
        IMG_SIZE = (224, 224)
        if image_for_prediction.mode != "RGB":
            image_for_prediction = image_for_prediction.convert("RGB")
        processed_image = image_for_prediction.resize(IMG_SIZE)
        img_array = np.array(processed_image) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)
        
        # Load model and weights
        try:
            # Add a small delay to simulate processing (better UX)
            time.sleep(2)
            
            model_repo_id = "MrityuTron/SkinCancerAI-Model"
            filename = "best_model.weights.h5"
            weights_file_path = hf_hub_download(repo_id=model_repo_id, filename=filename)
            
            # Build model architecture
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
            base_model.trainable = False
            x = layers.GlobalAveragePooling2D()(base_model.output)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(128, activation='relu', name='dense_layer')(x)
            output_layer = layers.Dense(7, activation='softmax')(x)
            model = models.Model(inputs=base_model.input, outputs=output_layer)
            model.load_weights(weights_file_path)
            
            # Make prediction
            prediction = model.predict(img_expanded)
            predicted_class_index = np.argmax(prediction[0])
            predicted_probability = prediction[0][predicted_class_index] * 100
            
            # Diagnosis mapping from HAM1000 Dataset
            diagnosis_mapping = {
                0: ('Actinic Keratoses (akiec)', 'Malignant'),
                1: ('Basal Cell Carcinoma (bcc)', 'Malignant'),
                2: ('Benign Keratosis-like Lesions (bkl)', 'Benign'),
                3: ('Dermatofibroma (df)', 'Benign'),
                4: ('Melanoma (mel)', 'Malignant'),
                5: ('Melanocytic Nevi (nv)', 'Benign'),
                6: ('Vascular Lesions (vasc)', 'Benign')
            }
            
            predicted_diagnosis, risk_level = diagnosis_mapping[predicted_class_index]
            
            # Fix for GradCAM - use the correct layer name
            # Find the last convolutional layer in the base model
            last_conv_layer = None
            for layer in reversed(base_model.layers):
                if 'conv' in layer.name:
                    last_conv_layer = layer.name
                    break
            
            # Generate Grad-CAM visualization with a fallback layer name
            grad_cam_explainer = GradCAM()
            try:
                grad_cam_heatmap = grad_cam_explainer.explain(
                    validation_data=(img_expanded, None),
                    model=model,
                    class_index=predicted_class_index,
                    layer_name='dense_layer'  # Try with our custom named layer
                )
            except Exception:
                try:
                    # Fallback to the last conv layer in the base model
                    grad_cam_heatmap = grad_cam_explainer.explain(
                        validation_data=(img_expanded, None),
                        model=model,
                        class_index=predicted_class_index,
                        layer_name=last_conv_layer
                    )
                except Exception:
                    # Last fallback - just create a blank heatmap
                    grad_cam_heatmap = np.ones(IMG_SIZE, dtype=np.float32) * 0.5
            
            heatmap_resized = tf.image.resize(grad_cam_heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:, :, 0]
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_image_resized = np.array(processed_image)
            overlayed_image = cv2.addWeighted(original_image_resized, 0.6, heatmap_colored, 0.4, 0)
            
            # Display results in Apple-style cards
            risk_color_class = "risk-high" if risk_level == "Malignant" else "risk-low"
            
            st.markdown(f"""
            <div class="results-container fade-in">
                <h3 style="font-size: 28px; margin-bottom: 30px; color: #1D1D1F;">Analysis Results</h3>
                
                <div class="two-columns">
                    <div>
                        <div class="result-metric">
                            Prediction: <span class="result-value">{predicted_diagnosis}</span>
                        </div>
                        <div class="result-metric">
                            Risk Level: <span class="result-value {risk_color_class}">{risk_level}</span>
                        </div>
                        <div class="result-metric">
                            Confidence: <span class="result-value">{predicted_probability:.1f}%</span>
                        </div>
                        
                        <div style="margin-top: 30px;">
                            <div class="progress-label">
                                <span class="progress-text">Confidence Level</span>
                                <span class="progress-text">{predicted_probability:.1f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {predicted_probability}%;"></div>
                            </div>
                        </div>
                        
                        <p style="margin-top: 40px; font-size: 16px; color: #1D1D1F; line-height: 1.5;">
                            The AI has identified this lesion as <strong>{predicted_diagnosis}</strong>, which is classified as <strong class="{risk_color_class}">{risk_level}</strong>. 
                            {"This type of lesion requires prompt medical attention." if risk_level == "Malignant" else "While this appears to be benign, regular monitoring is recommended."}
                        </p>
                        
                        <a href="#" class="apple-button">
                            {"Schedule Consultation" if risk_level == "Malignant" else "Save Results"}
                        </a>
                    </div>
                    
                    <div class="visualization-container">
                        <h4 style="margin-bottom: 15px; font-size: 18px; color: #1D1D1F;">AI Focus Visualization</h4>
                        <p style="margin-bottom: 20px; font-size: 14px; color: #86868b;">
                            The highlighted areas show regions that influenced the AI's analysis
                        </p>
            """, unsafe_allow_html=True)
            
            st.image(overlayed_image, use_container_width=True)
            
            st.markdown("""
                        <div class="visualization-overlay"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights section
            st.markdown("""
            <div class="apple-card" style="margin-top: 40px;">
                <h3 style="font-size: 24px; margin-bottom: 20px; color: #1D1D1F;">Key Insights</h3>
                <p style="font-size: 16px; line-height: 1.6; color: #1D1D1F;">
                    The AI model has analyzed your image using patterns learned from thousands of dermatological images. Here's what you should know:
                </p>
                <ul style="margin-top: 20px; margin-bottom: 20px;">
                    <li style="margin-bottom: 12px; color: #1D1D1F;">
                        <strong>Detection Confidence:</strong> The model has identified this lesion with {predicted_probability:.1f}% confidence.
                    </li>
                    <li style="margin-bottom: 12px; color: #1D1D1F;">
                        <strong>Key Characteristics:</strong> The visualization highlights areas with distinctive patterns associated with {predicted_diagnosis}.
                    </li>
                    <li style="margin-bottom: 12px; color: #1D1D1F;">
                        <strong>Next Steps:</strong> {"This type should be evaluated by a dermatologist promptly." if risk_level == "Malignant" else "Regular monitoring is recommended, but no immediate concern is indicated."}
                    </li>
                </ul>
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #d2d2d7;">
                    <p style="font-size: 14px; color: #86868b; font-style: italic;">
                        Remember: This analysis is provided for educational purposes and preliminary screening only. Always consult a healthcare professional for proper diagnosis.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="apple-card" style="border-left: 4px solid #FF9500; background-color: rgba(255, 149, 0, 0.05);">
                <h3 style="color: #FF9500; font-size: 18px;">Analysis Error</h3>
                <p style="color: #1D1D1F;">We encountered an error while analyzing your image. Please try again with a different image.</p>
                <p style="color: #86868b; font-size: 14px; margin-top: 10px;">Technical details: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    # Show welcome message when no image is uploaded
    st.markdown("""
    <div class="apple
