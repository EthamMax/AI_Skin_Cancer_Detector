import os
os.system("pip install huggingface-hub tf-explain opencv-python")  # Install required packages

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="SkinVision AI", layout="wide")

# --- Sidebar: About & Disclaimer ---
with st.sidebar:
    st.title("About SkinVision AI")
    st.markdown(
        """
        This web app uses a deep learning model to analyze skin lesion images and assess melanoma risk.
        It also provides **Grad-CAM visualization** to highlight the regions influencing the AI’s decision.
        
        **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.
        """
    )
    st.markdown("**Built by:** Mrityunjay Kumar")

# --- App Title and Instructions ---
st.title("SkinVision AI: Skin Cancer Detection")
st.markdown(
    """
    Upload a skin lesion image or capture a live photo to receive an AI-powered risk assessment along with a Grad-CAM overlay.
    """
)
st.subheader("Instructions for Best Results:")
st.markdown(
    """
    - **Ensure clarity:** The image should be clear, well-lit, and in focus.
    - **Close-up:** Capture the lesion as closely as possible with minimal background.
    - **Clean view:** Ensure the lesion is unobstructed by hair, bandages, or shadows.
    """
)

# --- Session State for Camera Activation ---
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# --- Layout: Two Columns for Input Methods ---
col1, col2 = st.columns(2)
image_for_prediction = None

# Column 1: File Upload
with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Select an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_for_prediction = Image.open(uploaded_file)
        st.image(image_for_prediction, caption="Uploaded Image", use_container_width=True)

# Column 2: Camera Input with Toggle
with col2:
    st.header("Capture a Live Photo")
    if not st.session_state.show_camera:
        if st.button("Activate Camera"):
            st.session_state.show_camera = True

    if st.session_state.show_camera:
        camera_image = st.camera_input("Take a live photo")
        if camera_image is not None:
            image_for_prediction = Image.open(camera_image)
            st.image(image_for_prediction, caption="Captured Live Photo", use_container_width=True)
            # Hide the camera widget after capture
            st.session_state.show_camera = False

# --- Process the Image if Available ---
if image_for_prediction is not None:
    # Standardized image size for model input and visualization
    IMG_SIZE = (224, 224)

    # Ensure image is in RGB mode
    if image_for_prediction.mode != "RGB":
        image_for_prediction = image_for_prediction.convert("RGB")
    
    # Preprocess image: resize and scale pixel values
    processed_image = image_for_prediction.resize(IMG_SIZE)
    img_array = np.array(processed_image) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)
    
    # Label mapping for diagnosis
    label_diagnosis_mapping = {
        0: 'Actinic Keratoses (akiec)',
        1: 'Basal Cell Carcinoma (bcc)',
        2: 'Benign Keratosis-like Lesions (bkl)',
        3: 'Dermatofibroma (df)',
        4: 'Melanoma (mel)',
        5: 'Melanocytic Nevi (nv)',
        6: 'Vascular Lesions (vasc)'
    }
    
    # --- Load Model Weights from Hugging Face Hub ---
    model_repo_id = "MrityuTron/SkinCancerAI-Model"
    filename = "best_model.weights.h5"
    weights_file_path = hf_hub_download(repo_id=model_repo_id, filename=filename)
    
    # --- Build the Model Architecture ---
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(7, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output_layer)
    
    # Load the pre-trained weights
    model.load_weights(weights_file_path)
    
    # --- Make Prediction ---
    prediction = model.predict(img_expanded)
    predicted_class_index = np.argmax(prediction[0])
    predicted_probability = prediction[0][predicted_class_index] * 100
    predicted_class_category = label_diagnosis_mapping[predicted_class_index]
    
    st.subheader("AI Analysis and Prediction")
    st.write(f"**Predicted Diagnosis:** {predicted_class_category}")
    st.write(f"**Confidence Level:** {predicted_probability:.2f}%")
    
    # --- Grad-CAM Visualization ---
    st.subheader("Grad-CAM Visualization")
    grad_cam_explainer = GradCAM()
    
    # Use an existing layer name; here we use 'out_relu' which exists in the model.
    grad_cam_heatmap = grad_cam_explainer.explain(
        validation_data=(img_expanded, None),
        model=model,
        class_index=predicted_class_index,
        layer_name='out_relu'
    )
    
    # Resize the Grad-CAM heatmap to the input image size
    heatmap_resized = tf.image.resize(grad_cam_heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:, :, 0]
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply a colormap for a colored heatmap (convert BGR to RGB for display)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Prepare the original (resized) image for overlay
    original_image_resized = np.array(processed_image)
    
    # Blend the original image with the heatmap
    overlayed_image = cv2.addWeighted(original_image_resized, 0.6, heatmap_colored, 0.4, 0)
    
    # Display the Grad-CAM overlay
    st.image(overlayed_image, caption=f"Grad-CAM Overlay (Prediction: {predicted_class_category})", use_container_width=False)

else:
    st.info("Please upload or capture an image to perform analysis.")
