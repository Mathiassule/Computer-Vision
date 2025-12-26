import streamlit as st
import cv2
import time
from datetime import datetime

st.set_page_config(page_title="Motion Detector", page_icon="🚨", layout="wide")
st.title("🚨 Intelligent Security Dashboard")

# --- CSS STYLING FOR ALARMS ---
# This injects custom HTML/CSS to make the status banners look professional
st.markdown("""
<style>
.safe {
    color: #28a745;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    border: 2px solid #28a745;
    border-radius: 10px;
    padding: 10px;
    background-color: #f0fff4;
}
.danger {
    color: #dc3545;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    border: 2px solid #dc3545;
    border-radius: 10px;
    padding: 10px;
    background-color: #ffe6e6;
    animation: blink 1s infinite; /* Flashing effect */
}
@keyframes blink {
  50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Control Panel")
run = st.sidebar.toggle('Activate System', value=False)
st.sidebar.caption("Adjust how big an object must be to trigger the alarm.")
min_area = st.sidebar.slider("Sensitivity (Area)", 100, 5000, 1000)

# --- DASHBOARD LAYOUT ---
# We use a placeholder so we can update the text without redrawing the whole page
status_placeholder = st.empty()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔴 Live Security Feed")
    FRAME_WINDOW = st.image([])
with col2:
    st.subheader("🕵️ Motion Mask")
    THRESH_WINDOW = st.image([])

# --- MAIN LOOP ---
camera = cv2.VideoCapture(0)

# Give the camera a moment to warm up
time.sleep(1) 

baseline_image = None

if run:
    while True:
        status = "Safe"
        ret, frame = camera.read()
        
        if not ret:
            st.error("Camera error. Please restart the app.")
            break

        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 2. Baseline Handling
        # The first frame we see becomes the "standard" for the room
        if baseline_image is None:
            baseline_image = gray
            continue

        # 3. Delta Calculation
        delta_frame = cv2.absdiff(baseline_image, gray)
        thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # 4. Contour Detection
        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            
            # Motion Detected!
            status = "Intruder Detected"
            (x, y, w, h) = cv2.boundingRect(contour)
            # Draw Red Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) 

        # 5. UI Updates
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        FRAME_WINDOW.image(frame_rgb, channels="RGB")
        THRESH_WINDOW.image(thresh_frame)

        # Update Status Banner
        if status == "Safe":
            status_placeholder.markdown('<div class="safe">✅ SYSTEM SAFE</div>', unsafe_allow_html=True)
        else:
            current_time = datetime.now().strftime("%H:%M:%S")
            status_placeholder.markdown(f'<div class="danger">⚠️ INTRUDER DETECTED at {current_time}</div>', unsafe_allow_html=True)

    camera.release()
else:
    status_placeholder.markdown('<div class="safe" style="color:gray; border-color:gray;">System Standby</div>', unsafe_allow_html=True)
    st.write("Toggle 'Activate System' in the sidebar to begin monitoring.")