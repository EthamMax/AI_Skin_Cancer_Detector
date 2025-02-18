import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from tensorflow.keras.applications import MobileNetV2 # Import MobileNetV2
from tensorflow.keras import layers, models # Import layers and models

# --- Streamlit App Header and Title ---
st.title("SkinVision AI: Skin Cancer Detection App") # Main title of the app
st.markdown("Upload a skin lesion image for AI-powered melanoma risk assessment.") # Subheading/introduction

# --- Sidebar for App Information and Instructions ---
with st.sidebar:
    st.title("About SkinVision AI")
    st.markdown("This web app uses a deep learning model to analyze skin lesion images and predict the likelihood of melanoma (a type of skin cancer).")
    st.markdown("It also provides a Grad-CAM visualization to explain the AI's decision.")
    st.markdown("Please note: This is a proof-of-concept tool for educational purposes and **not a substitute for professional medical advice.**")
    st.markdown("For any skin concerns, always consult a qualified dermatologist.")
    st.markdown("Built by: Mrityunjay Kumar") # Add your name here!


# --- Main App Content Area ---
st.header("Upload Your Skin Lesion Image") # Header for the image upload section

# --- Image Upload and Camera Input Options ---
uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"]) # File uploader
camera_image = st.camera_input("Or take a live photo using your camera") # Camera input

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
elif camera_image is not None:
    st.image(camera_image, caption="Live Photo from Camera.", use_column_width=True)


# --- Prediction and Grad-CAM Visualization Sections (To be implemented later) ---
# Sections to display AI prediction and Grad-CAM heatmap will go here

# --- Load Model Weights from Local File Path (in GitHub repo) ---
local_weights_file_path = "best_model.weights.h5" # Path to your model weights file (in the same directory as app.py in GitHub repo)

# --- Model Architecture (Code from Step 3.5 - CORRECTLY PLACED HERE, REPLACING PLACEHOLDER) ---
IMG_SIZE = (224, 224) # Define IMG_SIZE here as well, to be accessible in app.py
base_model = MobileNetV2(
    weights='imagenet',      
    include_top=False,       
    input_shape=IMG_SIZE + (3,) 
)
base_model.trainable = False 
global_average_pooling = layers.GlobalAveragePooling2D()(base_model.output) 
dropout_layer = layers.Dropout(0.5)(global_average_pooling) 
dense_layer_1 = layers.Dense(128, activation='relu')(dropout_layer) 
output_layer = layers.Dense(7, activation='softmax')(dense_layer_1) 
model = models.Model(inputs=base_model.input, outputs=output_layer)
# --- End of Model Architecture Code ---

model.load_weights(local_weights_file_path) # Load weights from local file

print(f"Model weights loaded from local file: {local_weights_file_path}") # Confirmation message
print("Streamlit app structure, image input widgets, and model loading from local file (GitHub repo) set up in app.py") # Confirmation message
