import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Background Remover", page_icon="✂️", layout="wide")

st.title("✂️ AI Background Remover")
st.markdown("**Week 10, Day 1: Visualizing the Segmentation Mask**")

# --- MEDIAPIPE SETUP ---
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Source", ["Live Webcam", "Image Upload"])
threshold = st.sidebar.slider("Mask Threshold", 0.1, 0.9, 0.5, help="Higher = stricter cut (less background). Lower = keeps more edges.")

# --- CORE LOGIC ---
def process_segmentation(image_input):
    # Initialize MediaPipe Selfie Segmentation
    # model_selection=1 is landscape mode (better for webcams)
    # model_selection=0 is general/square
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        # Convert to RGB
        if isinstance(image_input, np.ndarray):
             image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
             image_input = np.array(image_input)

        # 1. Run Inference
        results = selfie_segmentation.process(image_input)
        
        # 2. Get the Mask
        # results.segmentation_mask is a float array from 0.0 to 1.0
        mask = results.segmentation_mask
        
        # 3. Process Mask for Display
        # Create a boolean mask based on user threshold
        condition = np.stack((mask,) * 3, axis=-1) > threshold
        
        # Visualize:
        # Create a Black & White image to show the user "What the AI sees"
        # We multiply the mask (0-1) by 255 to get grayscale (0-255)
        visual_mask = (mask * 255).astype(np.uint8)
        
        # Make it 3 channels so Streamlit can display it easily as an image
        visual_mask_rgb = cv2.cvtColor(visual_mask, cv2.COLOR_GRAY2RGB)

        return visual_mask_rgb

# --- APP MODES ---
if mode == "Live Webcam":
    st.write("### 📹 Live Feed vs. AI Mask")
    run = st.checkbox('Start Camera')
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original Feed")
        FRAME_WINDOW_ORIG = st.image([])
    with col2:
        st.caption("Segmentation Mask (AI Vision)")
        FRAME_WINDOW_MASK = st.image([])
    
    camera = cv2.VideoCapture(0)
    
    if run:
        while True:
            ret, frame = camera.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            
            # Process
            mask_view = process_segmentation(frame)
            
            # Display
            FRAME_WINDOW_ORIG.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            FRAME_WINDOW_MASK.image(mask_view)
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
            mask_view = process_segmentation(image)
            st.image(mask_view, caption="The Mask")