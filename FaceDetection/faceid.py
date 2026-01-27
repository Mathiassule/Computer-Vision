import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="FaceID Database", page_icon="👤", layout="wide")

st.title("👤 FaceID: Database Manager")
st.markdown("**Week 9, Day 2: Building the Known Persons List**")

# --- DATABASE MANAGEMENT ---
DB_FILE = "face_db.pkl"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(database):
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)

# Load DB on startup
if 'face_db' not in st.session_state:
    st.session_state['face_db'] = load_db()

# --- SIDEBAR STATS ---
st.sidebar.header("🗂️ System Stats")
st.sidebar.metric("Registered People", len(st.session_state['face_db']))
if st.sidebar.button("Clear Database"):
    st.session_state['face_db'] = {}
    save_db({})
    st.rerun()

# --- CORE LOGIC ---
def generate_embedding(image):
    # DeepFace expects BGR
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        # Generate embedding
        results = DeepFace.represent(
            img_path = img_bgr, 
            model_name = "Facenet", 
            enforce_detection = True,
            detector_backend = "opencv"
        )
        # Return the first face found and its coordinates
        embedding = results[0]['embedding']
        area = results[0]['facial_area']
        return embedding, area, img_array
    
    except ValueError:
        return None, None, img_array
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

# --- MAIN UI TABS ---
tab1, tab2 = st.tabs(["➕ Add Person", "🔍 View Database"])

# --- TAB 1: REGISTRATION ---
with tab1:
    st.header("Register New Face")
    col1, col2 = st.columns(2)
    
    with col1:
        name_input = st.text_input("Enter Name", placeholder="e.g. Elon Musk")
        uploaded_file = st.file_uploader("Upload Clear Face Photo", type=['jpg', 'png', 'jpeg'])

    if uploaded_file and name_input:
        image = Image.open(uploaded_file).convert('RGB')
        
        with col2:
            st.image(image, caption="Preview", width=200)
            
        if st.button("Save to Database", type="primary"):
            with st.spinner(" extracting biometric signature..."):
                embedding, area, _ = generate_embedding(image)
                
                if embedding:
                    # Save to Session State and Disk
                    st.session_state['face_db'][name_input] = embedding
                    save_db(st.session_state['face_db'])
                    
                    st.success(f"✅ Successfully registered **{name_input}**!")
                    st.balloons()
                else:
                    st.error("❌ No face detected. Please use a clearer photo.")

# --- TAB 2: DATABASE INSPECTOR ---
with tab2:
    st.header("Registered Biometrics")
    
    if st.session_state['face_db']:
        # Convert dictionary to a nice format for display
        db_data = []
        for name, vector in st.session_state['face_db'].items():
            # Show first 5 numbers of the vector just to prove it exists
            preview_vector = [round(num, 4) for num in vector[:5]]
            db_data.append({"Name": name, "Vector Preview (First 5 dims)": str(preview_vector) + "..."})
        
        st.table(db_data)
    else:
        st.info("Database is empty. Go to 'Add Person' to register someone.")