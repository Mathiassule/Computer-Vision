import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import tempfile
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Retail Analytics", page_icon="🛒", layout="wide")

st.title("🛒 Smart Retail Analytics")
st.markdown("**Week 12, Day 2: The Footfall Counter**")

# --- MODEL LOADER ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Model Settings")
confidence = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.3, 0.05)

st.sidebar.divider()
st.sidebar.header("📏 Tripwire Settings")
st.sidebar.info("Adjust these sliders to place the line across the entrance.")

# Default line positions (relative 0.0 to 1.0)
start_x = st.sidebar.slider("Line Start X", 0.0, 1.0, 0.0)
start_y = st.sidebar.slider("Line Start Y", 0.0, 1.0, 0.5)
end_x = st.sidebar.slider("Line End X", 0.0, 1.0, 1.0)
end_y = st.sidebar.slider("Line End Y", 0.0, 1.0, 0.5)

# --- MAIN APP ---
uploaded_video = st.file_uploader("Upload Store Footage (MP4)", type=['mp4'])

if uploaded_video:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    col1, col2 = st.columns(2)
    
    # --- SETUP SUPERVISION ---
    # Get video info for scaling coordinates
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height = video_info.width, video_info.height
    
    # Create LineZone coordinates based on sliders
    line_start = sv.Point(int(start_x * width), int(start_y * height))
    line_end = sv.Point(int(end_x * width), int(end_y * height))
    
    # Initialize the Counting Zone
    line_zone = sv.LineZone(start=line_start, end=line_end)
    
    # Initialize Annotators (Updated for Supervision 0.15+)
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.8
    )
    # BoxAnnotator draws the bounding boxes
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    # LabelAnnotator draws the text/ID
    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.5
    )

    with col1:
        st.subheader("Setup Preview")
        # Show one frame with the line drawn so user can adjust
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            # Draw simple line for preview
            cv2.line(frame, (line_start.x, line_start.y), (line_end.x, line_end.y), (0, 0, 255), 4)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Adjust sliders to cross the path of walking people.")
        cap.release()

    # --- PROCESSING ---
    if st.button("Start Counting", type="primary"):
        cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()
        
        # Dashboard placeholders
        m1, m2 = st.columns(2)
        metric_in = m1.empty()
        metric_out = m2.empty()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # 1. Run Tracking
            results = model.track(frame, persist=True, conf=confidence, classes=[0], verbose=False)
            
            # 2. Check if we have tracks (Crucial check)
            if results[0].boxes.id is not None:
                # Convert to Supervision Detections
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Check for tracker IDs specifically
                if detections.tracker_id is not None:
                    # 3. Trigger Line Zone
                    line_zone.trigger(detections=detections)
                    
                    # 4. Annotate Frame
                    # Draw Boxes
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    
                    # Draw Labels (IDs)
                    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                    
            # Draw Line Counter (Always draw this, even if no detections, so we see the line)
            line_zone_annotator.annotate(frame=frame, line_counter=line_zone)
            
            # 5. Update Metrics
            metric_in.metric("⬇️ Entered", line_zone.in_count)
            metric_out.metric("⬆️ Exited", line_zone.out_count)
            
            # 6. Display
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        cap.release()
        st.success(f"Final Count: {line_zone.in_count} In, {line_zone.out_count} Out")

else:
    st.info("Upload a video to start.")