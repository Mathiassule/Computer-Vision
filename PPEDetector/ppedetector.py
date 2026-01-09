import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PPE Safety Dashboard", page_icon="🏗️", layout="wide")

st.title("🏗️ AI Safety Guard: Construction Site Monitor")
st.markdown("**Week 6: Custom PPE Detection & Compliance**")

# --- MODEL LOADER ---
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error("❌ 'best.pt' not found! Please place your trained model in this folder.")
        return None
    return YOLO(model_path)

model = load_model()

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ System Config")
confidence = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

if model:
    class_names = list(model.names.values())
    
    st.sidebar.divider()
    st.sidebar.header("📋 Compliance Rules")
    
    # Smart Defaults based on common PPE dataset names
    default_safe = [name for name in class_names if 'helmet' in name.lower() or 'vest' in name.lower()]
    default_unsafe = [name for name in class_names if 'no' in name.lower() or 'head' in name.lower()]
    
    safe_classes = st.sidebar.multiselect("✅ Safe (Green)", class_names, default=default_safe)
    unsafe_classes = st.sidebar.multiselect("❌ Unsafe (Red)", class_names, default=default_unsafe)

# --- HELPER: DRAWING LOGIC ---
def process_frame(frame, model, conf_threshold, safe_list, unsafe_list):
    """
    Takes a raw frame, runs inference, draws boxes, and calculates stats.
    """
    results = model(frame, conf=conf_threshold)
    
    # Stats for this frame
    frame_stats = {"Safe": 0, "Unsafe": 0}
    
    # Draw on the frame
    # YOLO plot() is good, but we need custom colors for compliance logic
    # We work on a copy to avoid modifying the original if needed
    annotated_frame = frame.copy()
    
    boxes = results[0].boxes
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        # Determine Color
        color = (128, 128, 128) # Grey
        if cls_name in safe_list:
            color = (0, 255, 0) # Green (BGR for OpenCV)
            frame_stats["Safe"] += 1
        elif cls_name in unsafe_list:
            color = (0, 0, 255) # Red (BGR for OpenCV)
            frame_stats["Unsafe"] += 1
            
        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw Box & Label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        label = f"{cls_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return annotated_frame, frame_stats

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["📸 Image Audit", "🎥 Video Surveillance"])

# --- TAB 1: IMAGES ---
with tab1:
    uploaded_img = st.file_uploader("Upload Site Photo", type=['jpg', 'png', 'jpeg'])
    if uploaded_img and model:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_img)
        img_array = np.array(image)
        
        with col1:
            st.image(image, caption="Original", use_container_width=True)
            
        if st.button("Run Audit", type="primary"):
            # Process (Note: OpenCV expects BGR, Pillow gives RGB. We usually draw in BGR then convert back)
            # Let's keep it simple: Convert to BGR for OpenCV drawing functions
            frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            processed_bgr, stats = process_frame(frame_bgr, model, confidence, safe_classes, unsafe_classes)
            
            # Convert back to RGB for Streamlit display
            processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(processed_rgb, caption="Analyzed", use_container_width=True)
                
            # Metrics
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Safe Workers", stats["Safe"])
            m2.metric("Violations", stats["Unsafe"], delta_color="inverse")
            
            total = stats["Safe"] + stats["Unsafe"]
            rate = round((stats["Safe"]/total)*100, 1) if total > 0 else 0
            m3.metric("Compliance Rate", f"{rate}%")

# --- TAB 2: VIDEO ---
with tab2:
    st.info("Upload a short site clip (CCTV footage).")
    uploaded_video = st.file_uploader("Upload MP4", type=['mp4'])
    
    if uploaded_video and model:
        # Save uploaded video to temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        vf = cv2.VideoCapture(tfile.name)
        
        stframe = st.image([])
        st_stats = st.empty()
        
        if st.button("Start Video Analysis"):
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Process Frame
                processed_frame, stats = process_frame(frame, model, confidence, safe_classes, unsafe_classes)
                
                # Display
                # Frame is already BGR, convert to RGB
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb)
                
                # Live Stats
                total = stats["Safe"] + stats["Unsafe"]
                rate = round((stats["Safe"]/total)*100, 1) if total > 0 else 0
                
                st_stats.markdown(f"""
                ### 📡 Live Stats
                **Safe:** {stats['Safe']} | **Unsafe:** {stats['Unsafe']} | **Compliance:** {rate}%
                """)
                
            vf.release()