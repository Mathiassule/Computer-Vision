import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Security Guard", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Security Guard: YOLOv8 Dashboard")
st.markdown("**Object Detection & Automated Logging**")

# --- STATE MANAGEMENT ---
if 'alarm_history' not in st.session_state:
    st.session_state['alarm_history'] = []

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    with st.spinner("Initializing Neural Network..."):
        model = load_model()
        class_names = list(model.names.values())
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ System Config")
mode = st.sidebar.radio("Operation Mode:", ["Live Monitor", "Static Analysis"])
confidence = st.sidebar.slider("AI Confidence", 0.0, 1.0, 0.35, 0.05)

st.sidebar.divider()
st.sidebar.subheader("🎯 Detection Filter")
defaults = ["person", "cell phone"]
selected_names = st.sidebar.multiselect("Active Objects:", class_names, default=defaults)
selected_ids = [k for k, v in model.names.items() if v in selected_names]

st.sidebar.divider()
st.sidebar.subheader("🚨 Logic Rules")
use_alarm = st.sidebar.toggle("Arm Security System")

target_class = None
max_count = 0

if use_alarm and selected_names:
    target_class = st.sidebar.selectbox("Trigger Alarm On:", selected_names)
    max_count = st.sidebar.number_input(f"Max Allowed {target_class}(s)", min_value=0, value=0)
    st.sidebar.info(f"Alarm triggers if > {max_count} {target_class}(s) detected.")

# --- HELPER FUNCTIONS ---
def draw_alarm(frame, text):
    # Red Border
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 30)
    # Text Banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def log_event(obj_name, count):
    # Add event to history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['alarm_history'].append({
        "Timestamp": timestamp,
        "Type": "ALARM TRIGGER",
        "Object": obj_name,
        "Count": count
    })

# --- MAIN UI TABS ---
tab1, tab2 = st.tabs(["👁️ Monitor", "📜 Incident Log"])

# --- TAB 1: MONITOR ---
with tab1:
    if mode == "Static Analysis":
        uploaded_file = st.file_uploader("Upload Surveillance Photo", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="Input Feed", use_container_width=True)
            
            if st.button("Run Scan", type="primary"):
                results = model(image, conf=confidence, classes=selected_ids)
                res_plotted = results[0].plot()
                
                # Logic Check
                boxes = results[0].boxes
                count = 0
                for box in boxes:
                    if model.names[int(box.cls[0])] == target_class:
                        count += 1
                
                if use_alarm and target_class and count > max_count:
                    res_plotted = draw_alarm(res_plotted, f"BREACH: {count} {target_class}s detected")
                    log_event(target_class, count)
                    st.error("🚨 SECURITY BREACH DETECTED")
                
                with col2:
                    st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="AI Analysis", use_container_width=True)

    elif mode == "Live Monitor":
        run = st.checkbox('🔴 START SURVEILLANCE FEED')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        last_logged = 0 # Cooldown timer to prevent spamming logs
        
        if run:
            while True:
                ret, frame = camera.read()
                if not ret: break
                
                # Inference
                results = model(frame, conf=confidence, classes=selected_ids)
                annotated_frame = results[0].plot()
                
                # Logic Layer
                if use_alarm and target_class:
                    boxes = results[0].boxes
                    current_count = 0
                    for box in boxes:
                        if model.names[int(box.cls[0])] == target_class:
                            current_count += 1
                    
                    if current_count > max_count:
                        annotated_frame = draw_alarm(annotated_frame, f"ALARM: {current_count} {target_class}")
                        
                        # Log Logic (with 2-second cooldown)
                        if time.time() - last_logged > 2:
                            log_event(target_class, current_count)
                            last_logged = time.time()
                
                FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        else:
            camera.release()
            st.info("System Standby. Check box to arm camera.")

# --- TAB 2: INCIDENT LOG ---
with tab2:
    st.subheader("Security Incident History")
    
    if st.session_state['alarm_history']:
        df = pd.DataFrame(st.session_state['alarm_history'])
        # Show most recent first
        st.dataframe(df.iloc[::-1], use_container_width=True)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Log (CSV)",
            csv,
            "security_log.csv",
            "text/csv"
        )
    else:
        st.write("No incidents recorded yet.")