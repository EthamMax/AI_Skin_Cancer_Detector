
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download # Import huggingface_hub
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models # Import model building components
from tf_explain.core.grad_cam import GradCAM # Import Grad-CAM explainer - ADDED
import matplotlib.pyplot as plt # Import matplotlib for Grad-CAM overlay - ADDED
import io # Import io for handling image bytes


# --- Streamlit App Header and Title ---
st.title("SkinVision AI: Skin Cancer Detection App") # Main title of the app
st.markdown("Upload a skin lesion image for AI-powered melanoma risk assessment with Grad-CAM visualization.") # Updated subheading

# --- Sidebar for App Information and Instructions ---
with st.sidebar:
    st.title("About SkinVision AI")
    st.markdown("This web app uses a deep learning model to analyze skin lesion images and predict the likelihood of melanoma (a type of skin cancer).")
    st.markdown("It provides **Grad-CAM visualization** to explain the AI's decision, highlighting the areas of concern.") # Updated description to mention Grad-CAM
    st.markdown("Please note: This is a proof-of-concept tool for educational purposes and **not a substitute for professional medical advice.**")
    st.markdown("For any skin concerns, always consult a qualified dermatologist.")
    st.markdown("Built by: Mrityunjay Kumar") # Add your name here!


# --- Main App Content Area ---
st.header("Upload or Capture Skin Lesion Image for Analysis") # Updated header

st.subheader("Instructions for Best Image Capture") # Instructions Subheader
st.markdown(
    """
    To get the most accurate AI analysis, please ensure your skin lesion image is:
    *   **Clearly focused and well-lit.**
    *   **Close-up:** Capture the lesion closely, filling most of the image area.
    *   **Include a ruler or scale (optional):** For size reference (though not mandatory).
    *   **Clean and unobstructed:** Ensure the lesion is not covered by hair, bandages, or shadows.
    """
)

st.subheader("Choose Image Input Method:") # Subheader for input options

col1, col2 = st.columns(2) # Create two columns for layout

image_for_prediction = None # Initialize image_for_prediction # Initialize image_for_prediction

with col1:
    uploaded_file = st.file_uploader("Upload from Storage", type=["jpg", "jpeg", "png"], key="file_upload") # File uploader in column 1
    if uploaded_file is not None:
        image_for_prediction = Image.open(uploaded_file) # Open uploaded file as PIL Image
        st.image(image_for_prediction, caption="Uploaded Image.", use_column_width=True) # Display PIL Image
        print(f"Type of image_for_prediction: {type(image_for_prediction)}") # Debug print

with col2:
    take_photo_button = st.button("Take Live Photo", key="camera_button") # Button to activate camera
    if take_photo_button:
        camera_image = st.camera_input("", key="camera_input") # Camera input - now activated by button
        if camera_image:
            image_for_prediction = Image.open(camera_image) # Open camera image as PIL Image
            st.image(camera_image, caption="Live Photo from Camera.", use_column_width=True) # Display camera image
            print(f"Type of image_for_prediction: {type(camera_image)}") # Debug print


if image_for_prediction is not None: # Proceed with prediction and Grad-CAM only if image is loaded
    IMG_SIZE = (224, 224)  # Define IMG_SIZE here (was missing)
    label_diagnosis_mapping = { # Updated label_diagnosis_mapping with full names
        0: 'Actinic Keratoses (akiec)',
        1: 'Basal Cell Carcinoma (bcc)',
        2: 'Benign Keratosis-like Lesions (bkl)',
        3: 'Dermatofibroma (df)',
        4: 'Melanoma (mel)',
        5: 'Melanocytic Nevi (nv)',
        6: 'Vascular Lesions (vasc)'
    }

    # Preprocess the image for prediction - NOW should work correctly with PIL Image
    img_array = np.array(image_for_prediction.resize(IMG_SIZE)) / 255.0  # Resize and rescale - NOW should work correctly
    img_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension
    print(f"Type of img_expanded (preprocessed image): {type(img_expanded)}, shape: {img_expanded.shape}") # Debug print


    # --- Load Model Weights from Hugging Face Hub ---
    model_repo_id = "MrityuTron/SkinCancerAI-Model" # CORRECT Hugging Face Repo ID - Double Check! - CORRECTED TO YOUR REPO ID
    filename = "best_model.weights.h5" # Filename of your weights file in the Hugging Face repo

    weights_file_path = hf_hub_download(repo_id=model_repo_id, filename=filename) # Download weights from Hugging Face Hub
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

    model.load_weights(weights_file_path) # Load the downloaded weights into the model

    print("Streamlit app structure, image input, AI prediction logic and Grad-CAM set up in app.py") # Confirmation message

    # Make prediction
    prediction = model.predict(img_expanded)
    predicted_class_index = np.argmax(prediction[0])  # Get index of max probability
    predicted_probability = prediction[0][predicted_class_index] * 100
    predicted_class_category = label_diagnosis_mapping[predicted_class_index]

    st.header("AI Analysis and Prediction")
    st.write(f"Predicted Diagnosis: **{predicted_class_category}**") # Now shows full category name
    st.write(f"Confidence Level: **{predicted_probability:.2f}%**")

    # --- Grad-CAM Visualization ---
    st.subheader("Grad-CAM Visualization") # Grad-CAM Subheader
    from tensorflow.keras.preprocessing import image # Ensure image is imported here - CORRECTED LINE - IMPORT INSIDE IF BLOCK
    grad_cam_explainer = GradCAM()

    grad_cam_heatmap = grad_cam_explainer.explain(
        validation_data=(img_expanded, None),
        model=model,
        class_index=predicted_class_index,
        layer_name='out_relu'
    )

    # --- Manual Heatmap Overlay using Matplotlib and OpenCV ---
    # Resize heatmap to match original image size - Grayscale heatmap (2D shape: 224x224) - FINAL RESIZE CORRECTION
    heatmap_resized = tf.image.resize(grad_cam_heatmap[..., tf.newaxis], IMG_SIZE).numpy()[:,:,0] # Resize to 2D grayscale

    heatmap_resized_uint8 = np.uint8(255 * heatmap_resized) # Scale to 0-255
    heatmap_resized_clip = np.clip(heatmap_resized_uint8, 0, 255) # Clip values to 0-255 if needed
    heatmap_colored = plt.cm.jet(heatmap_resized_clip)[:, :, :3] # Apply a colormap (jet colormap) - ENSURE RGB (3 channels) - THIS IS ACTUALLY RGB, NOT GRAYSCALE!

    # Load and resize original image to grayscale AND to IMG_SIZE for overlay - CORRECTED ORIGINAL IMAGE PROCESSING
    original_image_resized = image_for_prediction.resize(IMG_SIZE) # Resize PIL Image to IMG_SIZE for overlay - USING image_for_prediction directly! - RESIZE PIL IMAGE
    original_image_array_gray_resized = np.array(original_image_resized.convert('L')) / 255.0 # Convert resized PIL Image to grayscale numpy array, rescale - RESIZED GRAYSCALE IMAGE

    # --- Manual Heatmap Overlay using Matplotlib and OpenCV - CORRECTED OVERLAY CODE - NO REDUNDANT GRAYSCALE CONVERSION - CORRECT DATA TYPES!
    heatmap_resized_uint8 = np.uint8(heatmap_resized) # Convert heatmap_resized to uint8 - DATA TYPE CORRECTION! - CONVERT HEATMAP TO uint8
    original_image_array_gray_resized_uint8 = np.uint8(255 * original_image_array_gray_resized) # Force convert original image to uint8 - EXTREME FIX - FORCE uint8 - CONVERT ORIGINAL IMAGE TO uint8

    # Overlay heatmap on original image using OpenCV - USING RESIZED GRAYSCALE IMAGE - CORRECTED OVERLAY CODE - NO REDUNDANT GRAYSCALE CONVERSION - CORRECT DATA TYPES! - USING uint8 IMAGES
    overlayed_image_gray = cv2.addWeighted(heatmap_resized_uint8, 0.5, original_image_array_gray_resized_uint8, 0.5, 0) # OpenCV for weighted addition - USING uint8 HEATMAP and RESIZED GRAYSCALE IMAGE - CORRECTED OVERLAY - uint8 IMAGES

    st.image(overlayed_image_gray, caption=f"Grad-CAM Heatmap (Grayscale) - Predicted: {predicted_class_category}", use_column_width=True) # Display Grad-CAM heatmap - DISPLAYING GRAYSCALE OVERLAY - CORRECT DISPLAY METHOD!
    plt.axis('off') # Hide axes for cleaner visualization
    st.pyplot(plt) # Use st.pyplot to display matplotlib plot in Streamlit - CORRECT DISPLAY METHOD!

    print("Grad-CAM heatmap generated and displayed (simplified grayscale overlay with matplotlib and OpenCV).") # Confirmation message

else:
    print("Please upload or capture a skin lesion image to see AI analysis and Grad-CAM visualization.") # Message when no image is uploaded
