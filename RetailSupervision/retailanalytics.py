import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Retail Analytics", page_icon="🛒", layout="wide")

st.title("🛒 Smart Retail Analytics Suite")
st.markdown("**Week 12, Day 5: The Capstone Project (Final Build)**")

# --- MODEL LOADER ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    with st.spinner("Loading AI Brain..."):
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
time_speedup = st.sidebar.slider("Time Speed Multiplier", 1, 60, 5, help="1 real sec = X simulated secs")

# --- MAIN APP ---
uploaded_video = st.file_uploader("Upload Store Footage (MP4)", type=['mp4'])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    # Application Tabs
    tab1, tab2, tab3 = st.tabs(["📹 Live Feed & Setup", "📊 Analytics Dashboard", "📥 Export Data"])
    
    # --- SETUP SUPERVISION ---
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height = video_info.width, video_info.height
    fps = video_info.fps
    
    # Tripwire Zone
    line_start = sv.Point(int(start_x * width), int(start_y * height))
    line_end = sv.Point(int(end_x * width), int(end_y * height))
    line_zone = sv.LineZone(start=line_start, end=line_end)
    
    # Annotators
    heat_map_annotator = sv.HeatMapAnnotator(
        position=sv.Position.BOTTOM_CENTER, opacity=0.5, radius=25, kernel_size=25
    )
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.8)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # --- TAB 1: LIVE VIEW ---
    with tab1:
        col_vid, col_metric = st.columns([2, 1])
        
        with col_metric:
            st.subheader("Configuration Preview")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.line(frame, (line_start.x, line_start.y), (line_end.x, line_end.y), (0, 0, 255), 4)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Adjust sliders in sidebar to fit the door")
            cap.release()

        with col_vid:
            st.subheader("CCTV Processing Feed")
            if st.button("Start Analysis Engine", type="primary", use_container_width=True):
                cap = cv2.VideoCapture(video_path)
                st_frame = st.empty()
                
                # Live Metrics Placholders
                m1, m2, m3 = st.columns(3)
                metric_in = m1.empty()
                metric_out = m2.empty()
                metric_time = m3.empty()
                
                traffic_log = []
                base_time = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
                frame_count = 0
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success: break
                    
                    frame_count += 1
                    
                    # Simulated Time Logic
                    seconds_elapsed = (frame_count / fps) * time_speedup
                    current_sim_time = base_time + timedelta(seconds=seconds_elapsed)
                    time_str = current_sim_time.strftime("%H:%M:%S")
                    
                    # 1. AI Tracking
                    results = model.track(frame, persist=True, conf=confidence, classes=[0], verbose=False)
                    
                    if results[0].boxes.id is not None:
                        detections = sv.Detections.from_ultralytics(results[0])
                        
                        if detections.tracker_id is not None:
                            prev_in, prev_out = line_zone.in_count, line_zone.out_count
                            
                            # 2. Zone Trigger
                            line_zone.trigger(detections=detections)
                            
                            # 3. Event Logging
                            if line_zone.in_count > prev_in:
                                traffic_log.append({"Time": current_sim_time, "Event": "In", "Count": 1})
                            if line_zone.out_count > prev_out:
                                traffic_log.append({"Time": current_sim_time, "Event": "Out", "Count": 1})
                            
                            # 4. Drawing Layers
                            if show_heatmap:
                                frame = heat_map_annotator.annotate(scene=frame, detections=detections)
                            if show_boxes:
                                frame = box_annotator.annotate(scene=frame, detections=detections)
                                labels = [f"#{t_id}" for t_id in detections.tracker_id]
                                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                            
                    # Always draw Tripwire
                    line_zone_annotator.annotate(frame=frame, line_counter=line_zone)
                    
                    # UI Updates
                    metric_in.metric("⬇️ Total Entered", line_zone.in_count)
                    metric_out.metric("⬆️ Total Exited", line_zone.out_count)
                    metric_time.metric("🕒 Simulated Time", time_str)
                    
                    st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                cap.release()
                st.success("✅ Analysis Complete! Check the Dashboard tab.")
                
                # Save Data to Session
                st.session_state['traffic_data'] = pd.DataFrame(traffic_log) if traffic_log else pd.DataFrame()

    # --- TAB 2: DASHBOARD ---
    with tab2:
        st.header("Business Intelligence Dashboard")
        
        if 'traffic_data' in st.session_state and not st.session_state['traffic_data'].empty:
            df = st.session_state['traffic_data'].copy()
            df = df.set_index('Time')
            
            # Key Performance Indicators (KPIs)
            st.subheader("Key Performance Indicators (KPIs)")
            kpi1, kpi2, kpi3 = st.columns(3)
            
            total_in = len(df[df['Event'] == 'In'])
            total_out = len(df[df['Event'] == 'Out'])
            current_occupancy = total_in - total_out
            
            kpi1.metric("Total Visitors Today", total_in)
            kpi2.metric("Current Store Occupancy", current_occupancy)
            
            # Calculate Peak Minute
            resampled = df.resample('1min').count()
            if not resampled.empty and resampled['Count'].max() > 0:
                peak_time = resampled['Count'].idxmax().strftime("%H:%M")
                kpi3.metric("Peak Traffic Time", peak_time)
            else:
                kpi3.metric("Peak Traffic Time", "N/A")
            
            st.divider()
            
            # Charts
            st.subheader("Traffic Flow Over Time")
            df_in = df[df['Event'] == 'In'].resample('1min').count()['Count']
            df_out = df[df['Event'] == 'Out'].resample('1min').count()['Count']
            chart_data = pd.DataFrame({"Entering": df_in, "Exiting": df_out}).fillna(0)
            
            st.line_chart(chart_data)
        else:
            st.info("Run the Analysis Engine in the 'Live Feed' tab to generate BI Data.")

    # --- TAB 3: EXPORT DATA ---
    with tab3:
        st.header("Database Export")
        st.write("Download raw event logs for external BI tools (Excel, Tableau, PowerBI).")
        
        if 'traffic_data' in st.session_state and not st.session_state['traffic_data'].empty:
            df_raw = st.session_state['traffic_data']
            st.dataframe(df_raw.sort_values('Time', ascending=False), use_container_width=True)
            
            # Convert to CSV
            csv = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Traffic Log (CSV)",
                data=csv,
                file_name=f"retail_traffic_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("No data available to export. Please run an analysis first.")

else:
    st.info("Upload a video to start.")