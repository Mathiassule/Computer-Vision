import streamlit as st
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="3D Depth Engine", page_icon="🌌", layout="wide")

st.title("🌌 3D Depth Engine")
st.markdown("**Week 13, Day 1: Monocular Depth Estimation**")

# --- MODEL LOADER ---
@st.cache_resource
def load_depth_model():
    # FIXED: The task name is just "depth-estimation"
    # We use Intel's DPT (MiDaS) model from Hugging Face
    return pipeline("depth-estimation", model="Intel/dpt-large")

try:
    with st.spinner("Downloading MiDaS 3D AI Brain... (This takes a minute on first run)"):
        depth_estimator = load_depth_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- CORE LOGIC ---
def process_depth(image):
    # 1. Run the AI pipeline
    result = depth_estimator(image)
    
    # 2. Extract the Depth PIL Image
    # The model returns a grayscale image where White = Close, Black = Far
    depth_image = result['depth']
    
    # 3. Add a Heatmap Color for better visualization
    # Convert PIL to Numpy array
    depth_array = np.array(depth_image)
    
    # Ensure it's in the right format for OpenCV (8-bit grayscale)
    if depth_array.dtype != np.uint8:
        depth_array = (depth_array / np.max(depth_array) * 255).astype(np.uint8)
        
    # Apply OpenCV color map (INFERNO looks like thermal depth)
    depth_colormap = cv2.applyColorMap(depth_array, cv2.COLORMAP_INFERNO)
    
    return depth_image, depth_colormap

# --- MAIN APP ---
st.info("Upload any standard 2D photo and watch the AI calculate the physical distance of every pixel.")

uploaded_file = st.file_uploader("Upload a Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Load and convert to RGB
    original_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Flat 2D Image")
        st.image(original_image, use_container_width=True)
        
    if st.button("Calculate 3D Depth", type="primary"):
        with st.spinner("Calculating the Z-Axis..."):
            raw_depth, colored_depth = process_depth(original_image)
            
            with col2:
                st.subheader("2. Raw Depth Map")
                st.caption("White = Close | Black = Far")
                st.image(raw_depth, use_container_width=True)
                
            with col3:
                st.subheader("3. Thermal Distance")
                st.caption("Mapped to Inferno colormap")
                # Convert BGR (OpenCV) back to RGB for Streamlit
                st.image(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB), use_container_width=True)
                
        st.success("Depth calculation complete! The AI has successfully interpreted the Z-axis.")