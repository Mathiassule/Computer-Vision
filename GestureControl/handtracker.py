import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math
import pyautogui

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hand Tracker", page_icon="✋", layout="wide")

st.title("✋ MediaPipe Hand Tracker")
st.markdown("**Week 7: Hand Tracking & Gesture Control**")

# --- MEDIAPIPE SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Settings")
input_source = st.sidebar.radio("Input Source", ["Live Webcam", "Image Upload"])
app_mode = st.sidebar.selectbox("App Mode", ["Finger Counter", "Volume Control 🤏", "Gesture Recognition 🤟", "Virtual Mouse 🖱️"])

min_conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.7)

# --- GLOBAL VARIABLES FOR MOUSE SMOOTHING ---
if 'plocX' not in st.session_state: st.session_state['plocX'] = 0
if 'plocY' not in st.session_state: st.session_state['plocY'] = 0

# --- HELPER 1: FINGER COUNTING ---
def count_fingers(hand_landmarks, handedness_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb Logic
    if handedness_label == 'Right':
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else: 
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # 4 Fingers Logic
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# --- HELPER 2: VOLUME LOGIC ---
def calculate_volume(image, hand_landmarks):
    h, w, c = image.shape
    x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
    x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
    
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)
    
    vol = np.interp(length, [30, 200], [0, 100])
    vol_bar = np.interp(length, [30, 200], [400, 150])
    
    cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(image, f'{int(vol)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    
    return image, int(vol)

# --- HELPER 3: GESTURE RECOGNITION ---
def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]: return "Fist 👊"
    elif fingers == [1, 1, 1, 1, 1]: return "High Five ✋"
    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]: return "Peace ✌️"
    elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]: return "Rock On 🤘"
    elif fingers == [1, 0, 0, 0, 0]: return "Thumbs Up 👍"
    else: return ""

# --- HELPER 4: VIRTUAL MOUSE ---
def virtual_mouse(image, hand_landmarks):
    wCam, hCam = 640, 480
    frameR = 100 # Frame Reduction (Padding)
    smoothening = 5
    
    # Get Screen Dimensions
    wScr, hScr = pyautogui.size()
    
    # 1. Get Index Finger Tip (8) coordinates
    x1, y1 = hand_landmarks.landmark[8].x * wCam, hand_landmarks.landmark[8].y * hCam
    
    # Draw Active Region (Where you should move your hand)
    cv2.rectangle(image, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    # 2. Check if finger is in the Active Region
    # Convert Coordinates (Cam Space -> Screen Space)
    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
    
    # 3. Smoothen Values (Prevent Jitter)
    clocX = st.session_state['plocX'] + (x3 - st.session_state['plocX']) / smoothening
    clocY = st.session_state['plocY'] + (y3 - st.session_state['plocY']) / smoothening
    
    # 4. Move Mouse
    try:
        pyautogui.moveTo(wScr - clocX, clocY) # wScr - X to flip tracking horizontally
    except:
        pass # Handle edge cases where mouse goes out of bounds
    
    st.session_state['plocX'], st.session_state['plocY'] = clocX, clocY
    
    # Visual Marker for Mouse
    cv2.circle(image, (int(x1), int(y1)), 15, (255, 0, 255), cv2.FILLED)
    
    # 5. CLICK MODE: Check distance between Index (8) and Middle (12)
    x2, y2 = hand_landmarks.landmark[12].x * wCam, hand_landmarks.landmark[12].y * hCam
    length = math.hypot(x2 - x1, y2 - y1)
    
    if length < 40:
        cv2.circle(image, (int(x1), int(y1)), 15, (0, 255, 0), cv2.FILLED)
        pyautogui.click()
        cv2.putText(image, "CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image, "Active"

# --- CORE PROCESSING ---
def process_hand(image_input):
    with mp_hands.Hands(
        static_image_mode=(input_source == "Image Upload"),
        max_num_hands=1, # Mouse mode works best with 1 hand
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf) as hands:

        if isinstance(image_input, np.ndarray):
             image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
             image_input = np.array(image_input)

        results = hands.process(image_input)
        annotated_image = image_input.copy()
        display_val = 0

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                fingers_list = count_fingers(hand_landmarks, handedness)
                
                # --- MODE SWITCHING ---
                if app_mode == "Finger Counter":
                    count = fingers_list.count(1)
                    display_val += count
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = annotated_image.shape
                    cv2.putText(annotated_image, f"{count}", (int(wrist.x * w), int(wrist.y * h)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                                
                elif app_mode == "Volume Control 🤏":
                    annotated_image, vol = calculate_volume(annotated_image, hand_landmarks)
                    display_val = vol
                
                elif app_mode == "Gesture Recognition 🤟":
                    gesture = detect_gesture(fingers_list)
                    display_val = gesture
                    if gesture:
                        cv2.putText(annotated_image, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                        
                elif app_mode == "Virtual Mouse 🖱️":
                    # Only map if Index Finger is Up
                    if fingers_list[1] == 1: 
                        annotated_image, status = virtual_mouse(annotated_image, hand_landmarks)
                        display_val = status
        
        return annotated_image, display_val

# --- APP MODES ---
if input_source == "Live Webcam":
    st.write("### 📹 Live Feed")
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    
    camera = cv2.VideoCapture(0)
    camera.set(3, 640) # Set Width
    camera.set(4, 480) # Set Height
    
    if run:
        while True:
            ret, frame = camera.read()
            if not ret: break
            # Don't flip here for Mouse mode to keep movement intuitive, or handle logic carefully
            # Usually for mouse, we want mirrored movement (move right -> cursor right)
            frame = cv2.flip(frame, 1)
            annotated_frame, val = process_hand(frame)
            FRAME_WINDOW.image(annotated_frame)
    else:
        camera.release()

elif input_source == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        annotated_image, val = process_hand(image)
        st.image(annotated_image, caption=f"Result: {val}")