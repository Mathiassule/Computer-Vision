import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Retail Analytics", page_icon="🛒", layout="wide")

st.title("🛒 Smart Retail Analytics")
st.markdown("**Week 12, Day 1: Customer Tracking System**")

# --- MODEL LOADER ---
@st.cache_resource
def load_model():
    # Load YOLOv8 Nano (Fastest)
    return YOLO('yolov8n.pt')

try:
    with st.spinner("Loading AI Brain..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.3, 0.05)
# Tracker Configuration
# BoT-SORT is accurate, ByteTrack is fast. We'll use BoT-SORT (default).
tracker_type = st.sidebar.radio("Tracker Type", ["bytetrack.yaml", "botsort.yaml"])

# --- MAIN APP ---
uploaded_video = st.file_uploader("Upload Store Footage (MP4)", type=['mp4'])

if uploaded_video:
    # 1. Save Video to Temp File (OpenCV needs a file path)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Input Video Preview")
        st.video(video_path)

    # 2. Process Video
    if st.button("Start Tracking Analysis", type="primary"):
        cap = cv2.VideoCapture(video_path)
        
        st_frame = st.empty()
        
        with col2:
            st.info("Live Tracking Feed")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # --- CORE LOGIC: TRACKING ---
                # persist=True is CRITICAL. It tells YOLO to remember IDs between frames.
                # classes=[0] restricts detection to 'Person' class only.
                results = model.track(
                    frame, 
                    persist=True, 
                    conf=confidence, 
                    tracker=tracker_type,
                    classes=[0],
                    verbose=False
                )
                
                # Visualize
                # plot() draws the boxes AND the IDs (e.g., "id: 1")
                annotated_frame = results[0].plot()
                
                # Display
                # Convert BGR (OpenCV) to RGB (Streamlit)
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            cap.release()
            st.success("Analysis Complete!")

else:
    st.info("Upload a video to begin. (Tip: Use a video with people walking)")