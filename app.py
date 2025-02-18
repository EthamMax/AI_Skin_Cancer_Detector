import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# --- Streamlit App Header and Title ---
st.title("SkinVision AI: Skin Cancer Detection App")
st.markdown("Upload a skin lesion image for AI-powered melanoma risk assessment.")

# --- Sidebar for App Information and Instructions ---
with st.sidebar:
    st.title("About SkinVision AI")
    st.markdown("This web app uses a deep learning model to analyze skin lesion images and predict the likelihood of melanoma (a type of skin cancer).")
    st.markdown("It also provides a Grad-CAM visualization to explain the AI's decision.")
    st.markdown("Please note: This is a proof-of-concept tool for educational purposes and **not a substitute for professional medical advice.**")
    st.markdown("For any skin concerns, always consult a qualified dermatologist.")
    st.markdown("Built by: Mrityunjay Kumar")

# --- Main App Content Area ---
st.header("Upload Your Skin Lesion Image")

# --- Image Upload and Camera Input Options ---
uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or take a live photo using your camera")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
elif camera_image is not None:
    image = camera_image
    st.image(image, caption="Live Photo from Camera.", use_column_width=True)

# --- Prediction and Grad-CAM Visualization Sections ---
if uploaded_file is not None or camera_image is not None:  # Check if image is uploaded or captured
    st.header("AI Analysis and Prediction")  # Header for prediction section

    if uploaded_file is not None:
        # Correctly open UploadedFile as PIL Image - CORRECTED
        image_for_prediction = Image.open(uploaded_file)
    elif camera_image is not None:
        image_for_prediction = camera_image

    IMG_SIZE = (224, 224)  # Define IMG_SIZE here (was missing)
    label_diagnosis_mapping = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'} # ADD label_diagnosis_mapping here

    # Preprocess the image for prediction
    img_array = np.array(image_for_prediction.resize(IMG_SIZE)) / 255.0  # Resize and rescale - NOW should work correctly
    img_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Load Model Weights from Local File Path (in GitHub repo) ---
    local_weights_file_path = "best_model.weights.h5"  # Path to your model weights file (in the same directory as app.py in GitHub repo)

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

    model.load_weights(local_weights_file_path)  # Load weights from local file
    print(f"Model weights loaded from local file: {local_weights_file_path}")

    # Make prediction
    prediction = model.predict(img_expanded)
    predicted_class_index = np.argmax(prediction[0])  # Get index of max probability - CORRECTED LINE - PREVIOUSLY [1]
    predicted_probability = prediction[0][predicted_class_index] * 100  # Probability - CORRECTED LINE - PREVIOUSLY [1]
    predicted_class_category = label_diagnosis_mapping[predicted_class_index]  # Get category name from mapping

    # Display prediction results
    st.write(f"**Predicted Diagnosis:** {predicted_class_category}")
    st.write(f"**Confidence Level:** {predicted_probability:.2f}%")

    # Placeholder for Grad-CAM visualization display (will be added in next step)
    st.subheader("Grad-CAM Visualization (Coming Soon)")  # Subheader for Grad-CAM section
    st.markdown("Grad-CAM heatmap visualization will be displayed here in the next step.")

print("Streamlit app structure, image input, AI prediction logic set up in app.py") # Confirmation message
