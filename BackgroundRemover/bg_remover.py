import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Background Remover", page_icon="✂️", layout="wide")

st.title("✂️ AI Background Remover")
st.markdown("**Week 10, Day 3: Virtual Reality (Alpha Blending)**")

# --- MEDIAPIPE SETUP ---
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Source", ["Live Webcam", "Image Upload"])
view_mode = st.sidebar.radio("View Mode", ["Raw Mask (B&W)", "Green Screen (Hard Cut)", "Virtual Background 🏖️"])

# Logic specific configs
if view_mode == "Green Screen (Hard Cut)":
    threshold = st.sidebar.slider("Cut Threshold", 0.1, 0.9, 0.5, help="Adjust for sharp edges.")
    bg_img = None
elif view_mode == "Virtual Background 🏖️":
    threshold = 0.5 # Not used for soft blend, but kept variable
    bg_file = st.sidebar.file_uploader("Upload Background Image", type=['jpg', 'png', 'jpeg'])
    if bg_file:
        bg_img = np.array(Image.open(bg_file).convert('RGB'))
    else:
        bg_img = None
else:
    threshold = 0.5
    bg_img = None

# --- CORE LOGIC ---
def process_segmentation(image_input, background_array=None):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        # 1. Convert to RGB if needed (OpenCV usually gives BGR, Streamlit PIL gives RGB)
        # We assume input is RGB for consistency within this function
        # Ensure input is numpy array
        if not isinstance(image_input, np.ndarray):
             image_input = np.array(image_input)

        # 2. Run Inference
        results = selfie_segmentation.process(image_input)
        mask = results.segmentation_mask
        
        # 3. Handle View Modes
        
        # MODE A: Raw Mask (B&W)
        if view_mode == "Raw Mask (B&W)":
            condition = np.stack((mask,) * 3, axis=-1) > 0.5
            return (condition * 255).astype(np.uint8)

        # MODE B: Green Screen (Hard Cut)
        elif view_mode == "Green Screen (Hard Cut)":
            condition = np.stack((mask,) * 3, axis=-1) > threshold
            bg_image = np.zeros(image_input.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0) # Green
            return np.where(condition, image_input, bg_image)

        # MODE C: Virtual Background (Soft Blend)
        elif view_mode == "Virtual Background 🏖️":
            if background_array is None:
                # Fallback: Grey background if no image uploaded
                background_array = np.zeros(image_input.shape, dtype=np.uint8)
                background_array[:] = (100, 100, 100)
            else:
                # Resize user's background image to match webcam frame size
                h, w, c = image_input.shape
                background_array = cv2.resize(background_array, (w, h))

            # Stack mask to 3 channels (for RGB math)
            mask_3d = np.stack((mask,) * 3, axis=-1)
            
            # THE MATH: Soft Alpha Blending
            # Image * Mask + Background * (1 - Mask)
            # We convert to float for precise math, then back to uint8
            input_float = image_input.astype(float)
            bg_float = background_array.astype(float)
            
            output = (input_float * mask_3d) + (bg_float * (1.0 - mask_3d))
            return output.astype(np.uint8)

# --- APP MODES ---
if mode == "Live Webcam":
    st.write("### 📹 Studio Feed")
    run = st.checkbox('Start Camera')
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        FRAME_WINDOW_ORIG = st.image([])
    with col2:
        st.caption("Processed")
        FRAME_WINDOW_PROC = st.image([])
    
    camera = cv2.VideoCapture(0)
    
    if run:
        while True:
            ret, frame = camera.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # OpenCV Webcam is BGR, Convert to RGB for processing & display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process
            output = process_segmentation(frame_rgb, bg_img)
            
            # Display
            FRAME_WINDOW_ORIG.image(frame_rgb)
            FRAME_WINDOW_PROC.image(output)
    else:
        camera.release()

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Your Photo", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")
            
        with col2:
            output = process_segmentation(image, bg_img)
            st.image(output, caption="Result")