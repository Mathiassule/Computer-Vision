import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Personal Trainer", page_icon="🏋️‍♂️", layout="wide")

st.title("🏋️‍♂️ AI Personal Trainer")
st.markdown("**Week 8, Day 3: The Rep Counter**")

# --- MEDIAPIPE SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Source", ["Live Webcam", "Image Upload"])
smooth = st.sidebar.checkbox("Smooth Landmarks", value=True)
detection_conf = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
tracking_conf = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5)

# --- HELPER: CALCULATE ANGLE ---
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- APP MODES ---
if mode == "Live Webcam":
    st.write("### 📹 Live Gym Feed")
    run = st.checkbox('Start Training Session')
    FRAME_WINDOW = st.image([])
    
    # --- VARIABLES FOR REP COUNTER ---
    counter = 0 
    stage = None # "up" or "down"

    camera = cv2.VideoCapture(0)
    
    # Initialize Pose ONLY ONCE before the loop for performance
    with mp_pose.Pose(min_detection_confidence=detection_conf, min_tracking_confidence=tracking_conf) as pose:
        if run:
            while True:
                ret, frame = camera.read()
                if not ret: break
                
                # 1. Preprocessing
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 2. Process
                results = pose.process(image)
                
                # 3. Re-color for drawing
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 4. Pose Logic
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates (Left Arm)
                    # shoulder = 11, elbow = 13, wrist = 15
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Calculate Angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Visualize Angle
                    h, w, _ = image.shape
                    cv2.putText(image, str(int(angle)), 
                                tuple(np.multiply(elbow, [w, h]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # --- DAY 3 LOGIC: REP COUNTER ---
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage =='down':
                        stage = "up"
                        counter += 1
                        # Optional: Print to console
                        print(f"Rep: {counter}")
                        
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # 5. Dashboard UI (The Box)
                # Draw a blue rectangle box for stats
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                
                # Rep Data
                cv2.putText(image, 'REPS', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Stage Data
                cv2.putText(image, 'STAGE', (65, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage if stage else "-", (60, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display
                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            camera.release()

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Fitness Photo", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Simple processing for static image (no counting)
        with mp_pose.Pose(static_image_mode=True) as pose:
            img_array = np.array(image)
            results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            annotated_image = img_array.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.image(annotated_image, caption="Pose Analysis")