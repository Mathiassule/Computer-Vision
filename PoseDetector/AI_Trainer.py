import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Personal Trainer", page_icon="🏋️‍♂️", layout="wide")

st.title("🏋️‍♂️ AI Personal Trainer: The Complete Gym")
st.markdown("**Week 8: Pose Estimation & Rep Counting**")

# --- MEDIAPIPE SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Source", ["Live Webcam", "Image Upload"])
exercise_choice = st.sidebar.selectbox("Select Exercise", ["Bicep Curl", "Squat", "Pushup"])

st.sidebar.divider()
smooth = st.sidebar.checkbox("Smooth Landmarks", value=True)
detection_conf = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
tracking_conf = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5)

# --- HELPER: CALCULATE ANGLE ---
def calculate_angle(a, b, c):
    """
    Calculates angle between three points a, b, c.
    b is the vertex.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- EXERCISE LOGIC ---
def get_coordinates(landmarks):
    """
    Extracts relevant joints for all exercises.
    """
    return {
        'shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        'elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        'wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
        'hip': [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        'knee': [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        'ankle': [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
    }

# --- APP MODES ---
if mode == "Live Webcam":
    st.write(f"### 📹 Live Feed: {exercise_choice}")
    run = st.checkbox('Start Training Session')
    FRAME_WINDOW = st.image([])
    
    # --- VARIABLES FOR REP COUNTER ---
    counter = 0 
    stage = None # "up" or "down"
    feedback = "Good Form"

    camera = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=detection_conf, min_tracking_confidence=tracking_conf) as pose:
        if run:
            while True:
                ret, frame = camera.read()
                if not ret: break
                
                # 1. Preprocessing
                frame = cv2.flip(frame, 1) # Mirror view
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # 2. Process
                results = pose.process(image)
                
                # 3. Re-color
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 4. Pose Logic
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    coords = get_coordinates(landmarks)
                    
                    # --- EXERCISE SWITCHER ---
                    main_angle = 0
                    form_angle = 0
                    
                    if exercise_choice == "Bicep Curl":
                        # Main: Elbow Angle (Shoulder-Elbow-Wrist)
                        main_angle = calculate_angle(coords['shoulder'], coords['elbow'], coords['wrist'])
                        # Form: Back Posture (Shoulder-Hip-Knee)
                        form_angle = calculate_angle(coords['shoulder'], coords['hip'], coords['knee'])
                        
                        # Logic
                        if form_angle < 160: feedback = "Straighten Back!"
                        else: feedback = "Good Form"
                        
                        if feedback == "Good Form":
                            if main_angle > 160: stage = "down"
                            if main_angle < 30 and stage == 'down':
                                stage = "up"
                                counter += 1

                    elif exercise_choice == "Squat":
                        # Main: Knee Angle (Hip-Knee-Ankle)
                        main_angle = calculate_angle(coords['hip'], coords['knee'], coords['ankle'])
                        # Form: Knee alignment isn't easy in 2D, let's track depth
                        # Optional: Back Angle (Shoulder-Hip-Knee) - ensure not leaning TOO forward
                        form_angle = calculate_angle(coords['shoulder'], coords['hip'], coords['knee'])
                        
                        # Logic
                        if form_angle < 70: feedback = "Keep Chest Up!"
                        else: feedback = "Good Form"
                        
                        if feedback == "Good Form":
                            if main_angle > 160: stage = "up"
                            if main_angle < 90 and stage == 'up': # < 90 means deep squat
                                stage = "down"
                                counter += 1

                    elif exercise_choice == "Pushup":
                        # Main: Elbow Angle (Shoulder-Elbow-Wrist)
                        main_angle = calculate_angle(coords['shoulder'], coords['elbow'], coords['wrist'])
                        # Form: Body Alignment (Shoulder-Hip-Ankle) - Must be straight plank
                        form_angle = calculate_angle(coords['shoulder'], coords['hip'], coords['ankle'])
                        
                        # Logic
                        if form_angle < 160: feedback = "Fix Plank!"
                        else: feedback = "Good Form"
                        
                        if feedback == "Good Form":
                            if main_angle > 160: stage = "up"
                            if main_angle < 90 and stage == 'up':
                                stage = "down"
                                counter += 1

                    # --- VISUALIZATION ---
                    h, w, _ = image.shape
                    
                    # Draw Angle Text at the Main Joint
                    joint_pos = coords['elbow'] if exercise_choice != "Squat" else coords['knee']
                    cv2.putText(image, str(int(main_angle)), 
                                tuple(np.multiply(joint_pos, [w, h]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # --- DASHBOARD UI ---
                # Rep Data Box
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                
                cv2.putText(image, 'REPS', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, 'STAGE', (65, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, stage if stage else "-", (60, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Feedback Box
                color = (0, 255, 0) if feedback == "Good Form" else (0, 0, 255)
                cv2.rectangle(image, (400, 0), (640, 73), color, -1)
                cv2.putText(image, 'FEEDBACK', (415, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, feedback, (410, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display
                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            camera.release()

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Fitness Photo", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        with mp_pose.Pose(static_image_mode=True) as pose:
            img_array = np.array(image)
            results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            annotated_image = img_array.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.image(annotated_image, caption="Pose Analysis")