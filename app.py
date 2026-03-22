import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# explanation utilities for localization
from src.explainability.grad_cam import make_gradcam_heatmap, overlay_heatmap

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="KidneyInsight | Tumor Detection", page_icon="🩺", layout="centered")

# --- LOAD MODEL ---
# Using @st.cache_resource ensures the model loads only once, keeping the app fast
@st.cache_resource
def load_model():
    # Replace with the actual path to your model if it's inside the src/model folder
    return tf.keras.models.load_model('best_model.h5') 

model = load_model()

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image):
    # Convert PIL Image to OpenCV format (numpy array)
    img_array = np.array(image.convert('RGB'))
    
    # Resize to match your model's expected input shape (e.g., 224x224)
    # Update these dimensions to match what your model was trained on!
    img_resized = cv2.resize(img_array, (224, 224)) 
    
    # keep a copy of the resized image for visualization
    original = img_resized.copy()

    # Normalize if your model requires it (e.g., dividing by 255.0)
    img_normalized = img_resized / 255.0
    
    # Expand dimensions to create a batch of 1: shape becomes (1, 224, 224, 3)
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return original, img_expanded

# --- UI LAYOUT ---
st.title("🩺 KidneyInsight: AI Tumor Detection")
st.markdown("""
Upload a Kidney CT Scan image below. The deep learning model will analyze the scan and predict the presence of a tumor.
""")

st.divider()

# File Uploader
uploaded_file = st.file_uploader("Upload CT Scan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Scan")
        image = Image.open(uploaded_file)
        st.image(image, caption="Input CT Scan", use_container_width=True)

    with col2:
        st.subheader("Analysis Result")
        
        # Add a button to trigger the prediction
        if st.button("Analyze Scan", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess and predict
                original_img, processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Assuming binary classification: 0 (Normal) and 1 (Tumor)
                # Adjust the logic based on your model's exact output layer
                confidence = prediction[0][0]
                
                if confidence > 0.5:
                    st.error(f"🚨 **Tumor Detected** (Confidence: {confidence * 100:.2f}%)")
                    st.warning("Please consult a medical professional for a detailed diagnosis.")

                    # compute Grad-CAM heatmap
                    heatmap = make_gradcam_heatmap(processed_image, model)
                    # heatmap comes small (e.g. 7x7) – resize to match img
                    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

                    # apply mask to ignore black background
                    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                    mask = (gray > 10).astype('uint8')
                    heatmap = heatmap * mask

                    # further restrict overlay to central kidney area
                    h, w = heatmap.shape
                    ell = np.zeros_like(heatmap, dtype='uint8')
                    # radius roughly covers kidneys in 224x224 frame
                    cv2.ellipse(ell, (w//2, h//2), (int(w*0.4), int(h*0.25)), 0, 0, 360, 1, -1)
                    heatmap = heatmap * ell

                    orig_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

                    # derive a crude bbox from heatmap activations
                    _, thresh = cv2.threshold((heatmap*255).astype('uint8'), 50, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bbox_coords = None
                    if contours:
                        # take largest contour
                        c = max(contours, key=cv2.contourArea)
                        x,y,wbox,hbox = cv2.boundingRect(c)
                        bbox_coords = (x, y, x+wbox, y+hbox)
                        # draw rectangle on original for visualization
                        cv2.rectangle(orig_bgr, (x,y), (x+wbox, y+hbox), (0,255,0), 2)

                    overlay_bgr = overlay_heatmap(orig_bgr, heatmap)
                    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                    st.subheader("Tumor localization")
                    st.image(overlay_rgb, caption="Grad-CAM overlay", use_container_width=True)

                    if bbox_coords is not None:
                        st.write(f"Bounding box (xmin,ymin,xmax,ymax): {bbox_coords}")
                else:
                    st.success(f"✅ **No Tumor Detected** (Confidence: {(1 - confidence) * 100:.2f}%)")
                    st.info("The scan appears normal.")