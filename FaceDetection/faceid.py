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
st.set_page_config(page_title="FaceID Gatekeeper", page_icon="🛡️", layout="wide")

st.title("🛡️ FaceID: The Gatekeeper")
st.markdown("**Week 9, Day 5: Biometric Security Dashboard**")

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

# --- SIDEBAR: CONTROL PANEL ---
st.sidebar.header("⚙️ Security Settings")
# Threshold: Lower = Stricter (Vectors must be closer to match)
security_threshold = st.sidebar.slider("Security Threshold", 5, 25, 12, help="Lower values make the system stricter. Higher values accept more variation.")

st.sidebar.divider()
st.sidebar.subheader("📡 Live Activity Feed")
if os.path.exists(LOG_FILE):
    df_feed = pd.read_csv(LOG_FILE)
    # Show last 5 entries, newest on top, only Name and Time
    st.sidebar.dataframe(df_feed.tail(5).iloc[::-1][['Name', 'Time']], hide_index=True, use_container_width=True)
else:
    st.sidebar.info("No activity yet.")

# --- CORE FUNCTIONS ---
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(LOG_FILE, index=False)
    
    df = pd.read_csv(LOG_FILE)
    already_present = df[(df['Name'] == name) & (df['Date'] == date_str)]
    
    if already_present.empty:
        new_entry = pd.DataFrame({
            "Name": [name], 
            "Date": [date_str], 
            "Time": [time_str],
            "Status": ["Authorized"]
        })
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
        return True, time_str
    else:
        return False, already_present.iloc[0]['Time']

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

def find_match(target_embedding, db, threshold):
    min_dist = float("inf")
    identity = "Unknown"
    
    for name, db_embedding in db.items():
        dist = np.linalg.norm(np.array(target_embedding) - np.array(db_embedding))
        if dist < min_dist:
            min_dist = dist
            if dist < threshold:
                identity = name
    return identity, min_dist

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚪 Gate Control", "👤 Employee Database", "➕ New Registration", "📜 Full Logs"])

# --- TAB 1: GATE CONTROL (Main Dashboard) ---
with tab1:
    col_cam, col_status = st.columns([1.5, 1])
    
    with col_cam:
        st.subheader("CCTV Feed")
        scan_file = st.file_uploader("Upload Snapshot", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
        
        if scan_file:
            image = Image.open(scan_file).convert('RGB')
            st.image(image, caption="Live Input", use_container_width=True)
            
            # --- PROCESS BUTTON ---
            if st.button("VERIFY IDENTITY", type="primary", use_container_width=True):
                with st.spinner("Analyzing Biometrics..."):
                    results, img_array = get_embedding(image)
                    
                    if results:
                        annotated_img = img_array.copy()
                        
                        for face in results:
                            # Use dynamic threshold from sidebar
                            identity, dist = find_match(face['embedding'], st.session_state['face_db'], security_threshold)
                            area = face['facial_area']
                            x, y, w, h = area['x'], area['y'], area['w'], area['h']
                            
                            if identity == "Unknown":
                                color = (255, 0, 0)
                                status_text = "DENIED"
                                # Visuals for Unknown
                                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
                                
                                with col_status:
                                    st.error("ACCESS DENIED")
                                    st.metric(label="Identity", value="Unknown Person")
                                    st.metric(label="Distance Score", value=f"{dist:.2f}", delta="-High Risk")
                                    st.write("⚠️ **Security Alert:** Intruder detected.")

                            else:
                                color = (0, 255, 0)
                                status_text = "GRANTED"
                                # Log Logic
                                is_new, log_time = mark_attendance(identity)
                                
                                # Visuals for Known
                                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
                                
                                with col_status:
                                    st.success("ACCESS GRANTED")
                                    st.metric(label="Identity", value=identity)
                                    st.metric(label="Match Confidence", value=f"{dist:.2f}", delta="Low Distance (Good)")
                                    if is_new:
                                        st.toast(f"Welcome {identity}!")
                                    else:
                                        st.info(f"Already logged at {log_time}")

                        # Update the image with boxes
                        st.image(annotated_img, caption="Analysis Overlay", use_container_width=True)
                    else:
                        st.warning("No face detected in the image.")

# --- TAB 2: DATABASE ---
with tab2:
    st.subheader("Registered Personnel")
    if st.session_state['face_db']:
        # Show stats
        st.metric("Total Employees", len(st.session_state['face_db']))
        # Show list
        st.json(list(st.session_state['face_db'].keys()))
        
        if st.button("Clear Database (Reset)"):
            st.session_state['face_db'] = {}
            save_db({})
            st.rerun()
    else:
        st.info("Database is empty.")

# --- TAB 3: REGISTRATION ---
with tab3:
    st.subheader("Onboard New User")
    with st.form("reg_form"):
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            reg_name = st.text_input("Full Name")
            reg_file = st.file_uploader("Profile Photo", type=['jpg', 'png', 'jpeg'])
        with col_reg2:
            if reg_file:
                st.image(reg_file, width=200)
        
        submit = st.form_submit_button("Save to System")
        
        if submit and reg_file and reg_name:
            reg_image = Image.open(reg_file).convert('RGB')
            with st.spinner("Encoding..."):
                results, _ = get_embedding(reg_image)
                if results:
                    st.session_state['face_db'][reg_name] = results[0]['embedding']
                    save_db(st.session_state['face_db'])
                    st.success(f"✅ User '{reg_name}' registered successfully.")
                else:
                    st.error("Face could not be processed. Use a clearer photo.")

# --- TAB 4: LOGS ---
with tab4:
    st.subheader("Security Access Log")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df.iloc[::-1], use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Audit Log", csv, "access_log.csv", "text/csv")
    else:
        st.info("No logs generated yet.")