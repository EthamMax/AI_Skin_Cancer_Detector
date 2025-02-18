import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from huggingface_hub import hf_hub_download # Import huggingface_hub
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models # Import model building components

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

# --- Image Upload and Camera Input Options - Processing and Display ---
uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"]) # File uploader
camera_image = st.camera_input("Or take a live photo using your camera") # Camera input

image_for_prediction = None # Initialize image_for_prediction # Initialize image_for_prediction

if uploaded_file is not None:
    image_for_prediction = Image.open(uploaded_file) # Open uploaded file as PIL Image
    st.image(image_for_prediction, caption="Uploaded Image.", use_column_width=True) # Display PIL Image
    print(f"Type of image_for_prediction: {type(image_for_prediction)}") # Debug print

elif camera_image is not None:
    image_for_prediction = Image.open(camera_image) # Open camera image as PIL Image
    st.image(camera_image, caption="Live Photo from Camera.", use_column_width=True) # Display camera image
    print(f"Type of image_for_prediction: {type(image_for_prediction)}") # Debug print


if image_for_prediction is not None: # Proceed with prediction only if image is loaded
    IMG_SIZE = (224, 224)  # Define IMG_SIZE here (was missing)
    label_diagnosis_mapping = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'} # Define label mapping here (was missing)

    # Preprocess the image for prediction - NOW should work correctly with PIL Image
    img_array = np.array(image_for_prediction.resize(IMG_SIZE)) / 255.0  # Resize and rescale - NOW should work correctly
    img_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print(f"Type of image_for_prediction: {type(image_for_prediction)}") # Debug print


    # --- Load Model Weights from Local File Path (in GitHub repo) ---
    local_weights_file_path = "best_model.weights.h5"  # Path to your model weights file in the repo
    print(f"Model weights loaded from local file: {local_weights_file_path}") # Confirmation message

    # --- Model Architecture (Code from Step 3.5 - CORRECTLY PLACED HERE) ---
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

    model.load_weights(local_weights_file_path) # Load the weights - NOW should load correctly

    print("Streamlit app structure, image input, AI prediction logic set up in app.py") # Confirmation message

    # Make prediction
    prediction = model.predict(img_expanded)
    predicted_class_index = np.argmax(prediction[0])  # Get index of max probability
    predicted_probability = prediction[0][predicted_class_index] * 100
    predicted_class_category = label_diagnosis_mapping[predicted_class_index]

    st.header("AI Analysis and Prediction")
    st.write(f"Predicted Diagnosis: **{predicted_class_category}**")
    st.write(f"Confidence Level: **{predicted_probability:.2f}%**")

    st.subheader("Grad-CAM Visualization (Coming Soon)") # Placeholder for Grad-CAM
    st.write("Grad-CAM heatmap visualization will be displayed here in the next step.")
