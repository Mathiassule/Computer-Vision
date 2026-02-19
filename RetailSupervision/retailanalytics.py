import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Retail Analytics", page_icon="🛒", layout="wide")

st.title("🛒 Smart Retail Analytics")
st.markdown("**Week 12, Day 4: The Analytics Dashboard**")

# --- MODEL LOADER ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ View Settings")
confidence = st.sidebar.slider("Model Confidence", 0.0, 1.0, 0.3, 0.05)
show_heatmap = st.sidebar.toggle("Show Heatmap 🔥", value=True)
show_boxes = st.sidebar.toggle("Show Tracking Boxes", value=True)

st.sidebar.divider()
st.sidebar.header("📏 Tripwire Settings")
start_x = st.sidebar.slider("Line Start X", 0.0, 1.0, 0.0)
start_y = st.sidebar.slider("Line Start Y", 0.0, 1.0, 0.5)
end_x = st.sidebar.slider("Line End X", 0.0, 1.0, 1.0)
end_y = st.sidebar.slider("Line End Y", 0.0, 1.0, 0.5)

st.sidebar.divider()
st.sidebar.header("🕒 Simulation Settings")
start_hour = st.sidebar.number_input("Store Opening Hour", 0, 23, 9)
time_speedup = st.sidebar.slider("Time Speed Multiplier", 1, 60, 1, help="1 second of video = X seconds of store time")

# --- MAIN APP ---
uploaded_video = st.file_uploader("Upload Store Footage (MP4)", type=['mp4'])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    # Tabs for Live View vs Analytics
    tab1, tab2 = st.tabs(["📹 Live Feed", "📊 Analytics Dashboard"])
    
    # --- SETUP SUPERVISION ---
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height = video_info.width, video_info.height
    fps = video_info.fps
    
    # Tripwire Zone
    line_start = sv.Point(int(start_x * width), int(start_y * height))
    line_end = sv.Point(int(end_x * width), int(end_y * height))
    line_zone = sv.LineZone(start=line_start, end=line_end)
    
    # --- ANNOTATORS ---
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        opacity=0.5,
        radius=25,
        kernel_size=25
    )
    
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.8)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # --- TAB 1: LIVE VIEW ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Configuration Preview")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.line(frame, (line_start.x, line_start.y), (line_end.x, line_end.y), (0, 0, 255), 4)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Tripwire Placement")
            cap.release()

        # Processing Button
        if st.button("Start Analysis", type="primary"):
            cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            metric_in = m1.empty()
            metric_out = m2.empty()
            metric_time = m3.empty()
            
            # Data Logging
            traffic_log = []
            base_time = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                
                # Calculate Simulated Time
                # Current Video Time (seconds) * Speedup
                seconds_elapsed = (frame_count / fps) * time_speedup
                current_sim_time = base_time + timedelta(seconds=seconds_elapsed)
                time_str = current_sim_time.strftime("%H:%M:%S")
                
                # 1. Tracking
                results = model.track(frame, persist=True, conf=confidence, classes=[0], verbose=False)
                
                if results[0].boxes.id is not None:
                    detections = sv.Detections.from_ultralytics(results[0])
                    
                    if detections.tracker_id is not None:
                        # Store previous counts to detect changes
                        prev_in = line_zone.in_count
                        prev_out = line_zone.out_count
                        
                        # 2. Update Logic
                        line_zone.trigger(detections=detections)
                        
                        # Log Events if count changed
                        if line_zone.in_count > prev_in:
                            traffic_log.append({"Time": current_sim_time, "Event": "In", "Count": 1})
                        if line_zone.out_count > prev_out:
                            traffic_log.append({"Time": current_sim_time, "Event": "Out", "Count": 1})
                        
                        # 3. Drawing Layers
                        if show_heatmap:
                            frame = heat_map_annotator.annotate(scene=frame, detections=detections)
                        
                        if show_boxes:
                            frame = box_annotator.annotate(scene=frame, detections=detections)
                            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
                            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                        
                # Draw Tripwire
                line_zone_annotator.annotate(frame=frame, line_counter=line_zone)
                
                # Update Metrics
                metric_in.metric("⬇️ Entered", line_zone.in_count)
                metric_out.metric("⬆️ Exited", line_zone.out_count)
                metric_time.metric("🕒 Store Time", time_str)
                
                st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            cap.release()
            st.success("Analysis Complete")
            
            # Save Data to Session State for Tab 2
            if traffic_log:
                st.session_state['traffic_data'] = pd.DataFrame(traffic_log)
            else:
                st.session_state['traffic_data'] = pd.DataFrame()

    # --- TAB 2: DASHBOARD ---
    with tab2:
        st.header("Traffic Analytics")
        
        if 'traffic_data' in st.session_state and not st.session_state['traffic_data'].empty:
            df = st.session_state['traffic_data']
            
            # Resample data to 1-minute or 10-minute intervals for plotting
            # Set Time as index
            df = df.set_index('Time')
            
            # Separate In and Out
            df_in = df[df['Event'] == 'In'].resample('1min').count()['Count']
            df_out = df[df['Event'] == 'Out'].resample('1min').count()['Count']
            
            # Combine
            chart_data = pd.DataFrame({
                "Entering": df_in,
                "Exiting": df_out
            }).fillna(0)
            
            st.subheader("Traffic Flow (Per Simulated Minute)")
            st.line_chart(chart_data)
            
            st.subheader("Raw Event Log")
            st.dataframe(df.reset_index().sort_values('Time', ascending=False), use_container_width=True)
            
        else:
            st.info("Run the Analysis in the 'Live Feed' tab first to generate data.")

else:
    st.info("Upload a video to start.")