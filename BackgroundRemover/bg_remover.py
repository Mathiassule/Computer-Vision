import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Background Remover", page_icon="✂️", layout="wide")

st.title("✂️ AI Background Remover")
st.markdown("**Week 10, Day 2: The Cutout (Green Screen)**")

# --- MEDIAPIPE SETUP ---
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Source", ["Live Webcam", "Image Upload"])
view_mode = st.sidebar.radio("View Mode", ["Raw Mask (B&W)", "Green Screen (Cutout)"])
threshold = st.sidebar.slider("Cut Threshold", 0.1, 0.9, 0.5, help="Adjust to trim edges.")

# --- CORE LOGIC ---
def process_segmentation(image_input):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        # 1. Convert to RGB
        if isinstance(image_input, np.ndarray):
             image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
             image_input = np.array(image_input)

        # 2. Run Inference
        results = selfie_segmentation.process(image_input)
        mask = results.segmentation_mask
        
        # 3. Handle View Modes
        
        # MODE A: Raw Mask (Day 1 Logic)
        if view_mode == "Raw Mask (B&W)":
            # Make the mask binary based on threshold
            condition = np.stack((mask,) * 3, axis=-1) > threshold
            return (condition * 255).astype(np.uint8)

        # MODE B: Green Screen (Day 2 Logic)
        elif view_mode == "Green Screen (Cutout)":
            # Create a 3-channel mask
            # We smooth the mask for better edges (Linear Interpolation)
            # Or use strict binary for sharp edges. Let's use strict for Day 2 to see the math.
            condition = np.stack((mask,) * 3, axis=-1) > threshold
            
            # Create the Background (Solid Green)
            bg_image = np.zeros(image_input.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0) # R=0, G=255, B=0 (Green)
            
            # THE MATH: (Image * Mask) + (Background * Inverse_Mask)
            # np.where(condition, TrueValue, FalseValue)
            output_image = np.where(condition, image_input, bg_image)
            
            return output_image

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
            
            # Process
            output = process_segmentation(frame)
            
            # Display
            FRAME_WINDOW_ORIG.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            FRAME_WINDOW_PROC.image(output)
    else:
        camera.release()

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload a Photo", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original")
            
        with col2:
            output = process_segmentation(image)
            st.image(output, caption="Result")