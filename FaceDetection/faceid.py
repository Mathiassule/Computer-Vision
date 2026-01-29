import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import pickle
import pandas as pd
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="FaceID Attendance", page_icon="👤", layout="wide")

st.title("👤 FaceID: Smart Attendance System")
st.markdown("**Week 9, Day 4: Automated Logging**")

# --- FILES MANAGEMENT ---
DB_FILE = "face_db.pkl"
LOG_FILE = "attendance.csv"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(database):
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

if 'face_db' not in st.session_state:
    st.session_state['face_db'] = load_db()

# --- ATTENDANCE LOGIC ---
def mark_attendance(name):
    """
    Logs the person's entry into a CSV file.
    Prevents duplicate entries for the same day.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Create file if it doesn't exist
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(LOG_FILE, index=False)
    
    # Load current log
    df = pd.read_csv(LOG_FILE)
    
    # Check if already logged TODAY
    # Filter: Name matches AND Date matches
    already_present = df[(df['Name'] == name) & (df['Date'] == date_str)]
    
    if already_present.empty:
        # Add new entry
        new_entry = pd.DataFrame({
            "Name": [name], 
            "Date": [date_str], 
            "Time": [time_str],
            "Status": ["Present"]
        })
        # concat instead of append (pandas 2.0 best practice)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
        return True, time_str
    else:
        # Return False (not new) and the original check-in time
        return False, already_present.iloc[0]['Time']

# --- CORE LOGIC: EMBEDDING ---
def get_embedding(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    try:
        results = DeepFace.represent(
            img_path = img_bgr, 
            model_name = "Facenet", 
            enforce_detection = True,
            detector_backend = "opencv"
        )
        return results, img_array
    except ValueError:
        return None, img_array
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# --- CORE LOGIC: MATCHING ---
def find_match(target_embedding, db, threshold=10):
    min_dist = float("inf")
    identity = "Unknown"
    
    for name, db_embedding in db.items():
        dist = np.linalg.norm(np.array(target_embedding) - np.array(db_embedding))
        if dist < min_dist:
            min_dist = dist
            if dist < threshold:
                identity = name
    return identity, min_dist

# --- MAIN UI TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Biometric Scanner", "➕ Register Face", "🗂️ Database", "📝 Attendance Log"])

# --- TAB 1: SCANNER ---
with tab1:
    st.header("Gate Entry")
    scan_file = st.file_uploader("CCTV Snapshot", type=['jpg', 'png', 'jpeg'], key="scanner")
    
    if scan_file:
        image = Image.open(scan_file).convert('RGB')
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Live Feed", use_container_width=True)
            
        if st.button("Verify Access", type="primary"):
            with st.spinner("Processing Biometrics..."):
                results, img_array = get_embedding(image)
                
                if results:
                    annotated_img = img_array.copy()
                    
                    for face in results:
                        identity, dist = find_match(face['embedding'], st.session_state['face_db'])
                        area = face['facial_area']
                        x, y, w, h = area['x'], area['y'], area['w'], area['h']
                        
                        if identity == "Unknown":
                            color = (255, 0, 0)
                            label = f"Unknown ({dist:.2f})"
                        else:
                            color = (0, 255, 0)
                            label = f"{identity}"
                            
                            # --- DAY 4 LOGIC: LOG ATTENDANCE ---
                            is_new, log_time = mark_attendance(identity)
                            if is_new:
                                st.toast(f"✅ Entry logged for {identity} at {log_time}")
                            else:
                                st.toast(f"ℹ️ {identity} already checked in at {log_time}")

                        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(annotated_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    with col2:
                        st.image(annotated_img, caption="Access Result", use_container_width=True)
                else:
                    st.warning("No face detected.")

# --- TAB 2: REGISTER ---
with tab2:
    st.header("Register Employee")
    reg_name = st.text_input("Full Name", placeholder="e.g. Tony Stark")
    reg_file = st.file_uploader("Reference Photo", type=['jpg', 'png', 'jpeg'], key="register")

    if reg_file and reg_name:
        image = Image.open(reg_file).convert('RGB')
        st.image(image, width=200)
        if st.button("Save to Database"):
            with st.spinner("Encoding..."):
                results, _ = get_embedding(image)
                if results:
                    st.session_state['face_db'][reg_name] = results[0]['embedding']
                    save_db(st.session_state['face_db'])
                    st.success(f"✅ Registered {reg_name}")
                else:
                    st.error("Face not clear enough.")

# --- TAB 3: DATABASE ---
with tab3:
    st.header("System Database")
    if st.session_state['face_db']:
        st.json(list(st.session_state['face_db'].keys()))
    else:
        st.info("No registered users.")

# --- TAB 4: LOGS ---
with tab4:
    st.header("Daily Attendance Report")
    
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        # Show newest first
        st.dataframe(df.iloc[::-1], use_container_width=True)
        
        col_down, col_clear = st.columns(2)
        with col_down:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report (CSV)", csv, "attendance_report.csv", "text/csv")
        with col_clear:
            if st.button("🗑️ Reset Logs"):
                os.remove(LOG_FILE)
                st.rerun()
    else:
        st.info("No attendance records found yet.")