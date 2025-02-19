import os
os.system("pip install huggingface-hub") # Force install huggingface-hub at app start
os.system("pip install tf-explain") # Force install tf-explain at app start
os.system("pip install opencv-python") # Force install opencv-python at app start - ADDED FORCE INSTALL for OpenCV

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download # Import huggingface_hub
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models # Import model building components
from tf_explain.core.grad_cam import GradCAM # Import Grad-CAM explainer
import matplotlib.pyplot as plt # Import matplotlib for Grad-CAM overlay
import io # Import io for handling image bytes

# --- Streamlit App Header and Title ---
st.title("SkinVision AI: Skin Cancer Detection Web App")
st.markdown("Interactive melanoma risk assessment with AI & Grad-CAM visualization.") # More engaging subheading

# --- Sidebar for App Information and Instructions ---
with st.sidebar:
    st.title("SkinVision AI - Info")
    st.markdown("This app uses AI to analyze skin lesion images for melanoma risk.")
    st.markdown("It highlights areas of concern using Grad-CAM visualization.")
    st.markdown("⚠️ **Important:** This is for demonstration only, not medical advice.")
    st.markdown("Consult a dermatologist for skin concerns.")
    st.markdown("Created by: Mrityunjay Kumar")

# --- Main App Content Area ---
st.header("Analyze Your Skin Lesion") # More action-oriented header

st.subheader("Capture or Upload Image")
st.markdown("Get started by uploading an image from storage or capturing a live photo.") # Clearer instructions

col1, col2 = st.columns([2,1]) # Adjust column widths for better layout

image_for_prediction = None # Initialize image_for_prediction and crop_coords
crop_coords = None

with col1:
    uploaded_file = st.file_uploader("Upload Image from Storage", type=["jpg", "jpeg", "png"], key="file_upload") # More descriptive label
    if uploaded_file is not None:
        image_for_prediction = Image.open(uploaded_file)
        st.image(image_for_prediction, caption="Uploaded Image", use_column_width=True) # Clearer caption
        print(f"Image uploaded from storage: {uploaded_file.name}")

with col2:
    take_photo_button = st.button("Capture Live Photo", key="camera_button") # More descriptive button label
    if take_photo_button:
        camera_image = st.camera_input("Take photo", key="camera_input") # More descriptive camera input label
        if camera_image is not None:
            image_for_prediction = Image.open(camera_image)
            st.image(camera_image, caption="Live Photo", use_column_width=True) # Clearer caption
            print("Live photo captured from camera")

st.write("OR") # Separator text

crop_button = None # Initialize crop_button outside conditional block

if image_for_prediction is not None: # Proceed with cropping and analysis if image is loaded
    st.subheader("Adjust Crop (Optional)") # Subheader for cropping

    # --- Image Cropping UI (Basic Square Crop) ---
    left, top, right, bottom = 0, 0, image_for_prediction.width, image_for_prediction.height # Initial crop area - full image
    default_crop_size = min(image_for_prediction.width, image_for_prediction.height)
    right = default_crop_size
    bottom = default_crop_size

    crop_coords = st.slider("Adjust Crop Area", 
                            min_value=0, max_value=min(image_for_prediction.width, image_for_prediction.height), 
                            value=(0, default_crop_size), 
                            format="pixels", 
                            key="crop_slider") # Slider for crop size - basic square crop

    left, top = 0, 0 # Assume top-left corner is fixed for simplicity
    right = crop_coords[1]
    bottom = crop_coords[1]

    cropped_image_pil = image_for_prediction.crop((left, top, right, bottom)) # Apply crop using PIL
    st.image(cropped_image_pil, caption="Cropped Image (Optional)", use_column_width=True) # Display cropped image

    crop_button = st.button("Crop and Analyze", key="crop_analyze_button") # "Crop and Analyze" button


if crop_button: # If "Crop and Analyze" button is clicked, proceed with analysis on cropped image
    if cropped_image_pil:
        image_to_analyze = cropped_image_pil # Use cropped image for analysis
        st.write("Analyzing cropped image...") # Indicate analyzing cropped image
    else:
        image_to_analyze = image_for_prediction # If cropped image is None (should not happen), use original image as fallback
        st.write("Analyzing original image...") # Indicate analyzing original image

    if image_to_analyze:
        IMG_SIZE = (224, 224)
        label_diagnosis_mapping = { 0: 'Actinic Keratoses (akiec)', 1: 'Basal Cell Carcinoma (bcc)', 2: 'Benign Keratosis-like Lesions (bkl)', 3: 'Dermatofibroma (df)', 4: 'Melanoma (mel)', 5: 'Melanocytic Nevi (nv)', 6: 'Vascular Lesions (vasc)'}

        img_array = np.array(image_to_analyze.resize(IMG_SIZE)) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        model_repo_id = "MrityuTron/SkinCancerAI-Model" # CORRECT Hugging Face Repo ID
        filename = "best_model.weights.h5"

        weights_file_path = hf_hub_download(repo_id=model_repo_id, filename=filename)
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
        base_model.trainable = False
        global_average_pooling = layers.GlobalAveragePooling2D()(base_model.output)
        dropout_layer = layers.Dropout(0.5)(global_average_pooling)
        dense_layer_1 = layers.Dense(128, activation='relu')(dropout_layer)
        output_layer = layers.Dense(7, activation='softmax')(dense_layer_1)
        model = models.Model(inputs=base_model.input, outputs=output_layer)
        model.load_weights(weights_file_path)

        prediction = model.predict(img_expanded)
        predicted_class_index = np.argmax(prediction[0])
        predicted_probability = prediction[0][predicted_class_index] * 100
        predicted_class_category = label_diagnosis_mapping[predicted_class_index]

        st.header("AI Analysis and Prediction")
        st.write(f"Predicted Diagnosis: **{predicted_class_category}**")
        st.write(f"Confidence Level: **{predicted_probability:.2f}%**")

        # --- Grad-CAM Visualization ---
        st.subheader("Grad-CAM Visualization") # Updated Subheader - Grad-CAM is NOW here!
        from tensorflow.keras.preprocessing import image # Ensure image is imported here
        grad_cam_explainer = GradCAM()

        grad_cam_heatmap = grad_cam_explainer.explain(
            validation_data=(img_expanded, None),
            model=model,
            class_index=predicted_class_index,
            layer_name='out_relu'
        )

        heatmap_resized = tf.image.resize(grad_cam_heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:,:,0]
        heatmap_resized_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_resized_clip = np.clip(heatmap_resized_uint8, 0, 255)
        heatmap_colored = plt.cm.jet(heatmap_resized_clip)[:, :, :3]

        original_image_array_gray_resized = np.array(image_to_analyze.convert('L').resize(IMG_SIZE)) / 255.0 # Use cropped image for Grad-CAM, resize and grayscale

        overlayed_image_gray = cv2.addWeighted(heatmap_resized_uint8, 0.5, original_image_array_gray_resized[..., np.newaxis] * 0.5, 0.5, 0)

        st.image(overlayed_image_gray, caption=f"Grad-CAM Heatmap - Predicted: {predicted_class_category}", use_column_width=True) # Display Grad-CAM heatmap
        plt.axis('off')
        st.pyplot(plt)

        print("Grad-CAM heatmap generated and displayed (simplified grayscale overlay with matplotlib and OpenCV).")

    else:
        st.warning("Please upload or capture a skin lesion image to see AI analysis and Grad-CAM visualization.") # More informative warning message
else:
    st.write("Awaiting image upload or capture to enable crop and analysis.") # Message before image upload/capture

print("Streamlit app structure, image input, AI prediction logic and Grad-CAM (FINAL VERSION - WITH CROPPING UI) set up in app.py") # Confirmation message
