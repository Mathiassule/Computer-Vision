import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Background Remover Studio", page_icon="✂️", layout="wide")

st.title("✂️ AI Background Remover Studio")
st.markdown("**Week 10: Content Creator Suite**")

# --- MEDIAPIPE SETUP ---
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# --- SIDEBAR ---
st.sidebar.header("⚙️ Studio Settings")
mode = st.sidebar.radio("Input Source", ["Image Upload", "Live Webcam"])
view_mode = st.sidebar.radio("Effect Mode", [
    "Transparent (PNG Sticker) 🖼️",
    "Portrait Mode (Blur) 📷",
    "Virtual Background 🏖️",
    "Green Screen (Hard Cut)", 
    "Raw Mask (Debug)"
])

# Logic specific configs
if view_mode == "Green Screen (Hard Cut)":
    threshold = st.sidebar.slider("Cut Threshold", 0.1, 0.9, 0.5)
    bg_img = None
    blur_strength = 0
elif view_mode == "Virtual Background 🏖️":
    threshold = 0.5
    bg_file = st.sidebar.file_uploader("Upload Background Image", type=['jpg', 'png', 'jpeg'])
    if bg_file:
        bg_img = np.array(Image.open(bg_file).convert('RGB'))
    else:
        bg_img = None
    blur_strength = 0
elif view_mode == "Portrait Mode (Blur) 📷":
    threshold = 0.5
    bg_img = None
    blur_strength = st.sidebar.slider("Blur Intensity", 3, 99, 21, step=2)
else: # Transparent or Raw
    threshold = 0.5
    bg_img = None
    blur_strength = 0

# --- CORE LOGIC ---
def process_segmentation(image_input, background_array=None, blur=0):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        if not isinstance(image_input, np.ndarray):
             image_input = np.array(image_input)

        results = selfie_segmentation.process(image_input)
        mask = results.segmentation_mask
        
        # Stack mask for 3-channel math
        mask_3d = np.stack((mask,) * 3, axis=-1)

        # MODE 1: Transparent PNG (New for Day 5)
        if view_mode == "Transparent (PNG Sticker) 🖼️":
            # Create RGBA Image
            # RGB from original
            r, g, b = cv2.split(image_input)
            # Alpha from mask (float 0-1 -> uint8 0-255)
            a = (mask * 255).astype(np.uint8)
            # Merge into 4-channel image
            return cv2.merge((r, g, b, a))

        # MODE 2: Portrait Blur
        elif view_mode == "Portrait Mode (Blur) 📷":
            blurred_image = cv2.GaussianBlur(image_input, (blur, blur), 0)
            input_float = image_input.astype(float)
            blurred_float = blurred_image.astype(float)
            output = (input_float * mask_3d) + (blurred_float * (1.0 - mask_3d))
            return output.astype(np.uint8)

        # MODE 3: Virtual Background
        elif view_mode == "Virtual Background 🏖️":
            if background_array is None:
                background_array = np.zeros(image_input.shape, dtype=np.uint8)
                background_array[:] = (100, 100, 100) # Grey default
            else:
                h, w, c = image_input.shape
                background_array = cv2.resize(background_array, (w, h))
            
            bg_float = background_array.astype(float)
            input_float = image_input.astype(float)
            output = (input_float * mask_3d) + (bg_float * (1.0 - mask_3d))
            return output.astype(np.uint8)

        # MODE 4: Green Screen
        elif view_mode == "Green Screen (Hard Cut)":
            condition = np.stack((mask,) * 3, axis=-1) > threshold
            bg_image = np.zeros(image_input.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0)
            return np.where(condition, image_input, bg_image)

        # MODE 5: Raw Mask
        elif view_mode == "Raw Mask (Debug)":
            condition = np.stack((mask,) * 3, axis=-1) > 0.5
            return (condition * 255).astype(np.uint8)

# --- APP MODES ---
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Your Photo", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
            
        with col2:
            with st.spinner("Processing..."):
                output = process_segmentation(image, bg_img, blur_strength)
                st.image(output, caption="Result", use_container_width=True)
                
                # Download Logic
                buf = io.BytesIO()
                # Determine format
                if view_mode == "Transparent (PNG Sticker) 🖼️":
                    save_fmt = "PNG"
                    mime_type = "image/png"
                    # PIL expects RGBA for PNG saving
                    Image.fromarray(output).save(buf, format=save_fmt)
                else:
                    save_fmt = "JPEG"
                    mime_type = "image/jpeg"
                    Image.fromarray(output).save(buf, format=save_fmt)
                
                st.download_button(
                    label=f"📥 Download {save_fmt}",
                    data=buf.getvalue(),
                    file_name=f"edited_image.{save_fmt.lower()}",
                    mime=mime_type
                )

elif mode == "Live Webcam":
    st.write("### 📹 Live Studio Feed")
    run = st.checkbox('Start Camera')
    
    col1, col2 = st.columns(2)
    with col1:
        FRAME_WINDOW_ORIG = st.image([])
    with col2:
        FRAME_WINDOW_PROC = st.image([])
    
    camera = cv2.VideoCapture(0)
    
    if run:
        while True:
            ret, frame = camera.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            output = process_segmentation(frame_rgb, bg_img, blur_strength)
            
            FRAME_WINDOW_ORIG.image(frame_rgb)
            FRAME_WINDOW_PROC.image(output)
    else:
        camera.release()