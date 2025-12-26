import streamlit as st
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import re
import pandas as pd
import os
import platform
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Scanner AI", page_icon="📜", layout="wide")

st.title("📜 Smart Scanner AI")
st.markdown("**Week 4: Optical Character Recognition & Data Mining**")

# --- ROBUST TESSERACT PATH FINDER ---
def find_tesseract_binary():
    """
    Actively searches for the Tesseract binary in common locations
    to avoid PATH errors on both Local Windows and Cloud Linux.
    """
    # 1. Check if it's already in the PATH (Best case)
    # shutil.which returns the full path if found, or None
    path_in_env = shutil.which("tesseract")
    if path_in_env:
        return path_in_env
    
    # 2. Common Windows Paths
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe")
    ]
    
    # 3. Common Linux/Mac Paths
    linux_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]
    
    # Check OS and search specific paths
    system = platform.system()
    search_paths = windows_paths if system == "Windows" else linux_paths
    
    for path in search_paths:
        if os.path.exists(path):
            return path
            
    return None

# --- SETUP TESSERACT ---
tesseract_path = find_tesseract_binary()

if tesseract_path:
    # Set the path explicitly
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # If we still can't find it, show a big helpful error
    st.error("🚨 Critical Error: Tesseract Engine not found!")
    
    # DIAGNOSTICS FOR DEBUGGING
    with st.expander("🔍 Click for Troubleshooting Info"):
        st.write(f"**OS detected:** {platform.system()}")
        st.write("Checked standard paths but found nothing.")
        
        if platform.system() == "Linux":
            st.warning("""
            **Streamlit Cloud Instructions:**
            1. Ensure you have a file named `packages.txt` in your repo root.
            2. It must contain the line: `tesseract-ocr`.
            3. **CRITICAL:** Go to 'Manage App' (bottom right) -> 3 dots -> **Reboot App**. 
            (Adding the file isn't enough; the server must restart to install it).
            """)
            
            # Check if /usr/bin exists to debug
            if os.path.exists("/usr/bin"):
                files = os.listdir("/usr/bin")
                tess_files = [f for f in files if "tess" in f]
                if tess_files:
                    st.success(f"Found similar files in /usr/bin: {tess_files}")
                    st.write("Try setting path manually in sidebar.")
                else:
                    st.error("No 'tesseract' files found in /usr/bin. It is NOT installed.")
    
    if platform.system() == "Windows":
        st.warning("It looks like you are on Windows. Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        st.info("If you installed it, paste the full path to 'tesseract.exe' in the sidebar.")

# --- SIDEBAR: MANUAL OVERRIDE (Safety Net) ---
# If the auto-finder fails, let the user paste the path manually
if not tesseract_path:
    manual_path = st.sidebar.text_input("Manual Tesseract Path:", placeholder="C:\Program Files\Tesseract-OCR\tesseract.exe")
    if manual_path:
        pytesseract.pytesseract.tesseract_cmd = manual_path
        st.sidebar.success("Path updated!")

st.sidebar.header("🎛️ Image Preprocessing")
blur_amount = st.sidebar.slider("Denoise (Blur)", 1, 15, 5, step=2)
block_size = st.sidebar.slider("Shadow Threshold (Block)", 3, 51, 21, step=2)
c_const = st.sidebar.slider("Contrast (C)", 1, 30, 10)

# --- CORE FUNCTIONS ---

def preprocess_image(image, blur_val, block_val, c_val):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    gray = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    processed_img = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_val, 
        c_val 
    )
    return processed_img

def extract_text(image_data):
    try:
        text = pytesseract.image_to_string(image_data)
        return text
    except Exception as e:
        # Don't crash, just report
        return None

def parse_data(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    url_pattern = r'(?:https?://|www\.)[\w\.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\b0\d{10}\b'

    return {
        "Emails": re.findall(email_pattern, text),
        "Dates": re.findall(date_pattern, text),
        "Links": re.findall(url_pattern, text),
        "Phones": re.findall(phone_pattern, text)
    }

def search_and_highlight(processed_img, original_img, search_term):
    try:
        d = pytesseract.image_to_data(processed_img, output_type=Output.DICT)
        overlay_img = np.array(original_img)
        found_count = 0
        n_boxes = len(d['text'])
        
        for i in range(n_boxes):
            if int(d['conf'][i]) > 40:
                word = d['text'][i]
                if search_term.lower() in word.lower() and search_term.strip() != "":
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    found_count += 1
        return overlay_img, found_count
    except Exception as e:
        return np.array(original_img), 0

# --- MAIN APP UI ---
uploaded_file = st.file_uploader("Upload Document", type=["jpg", "png", "jpeg"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    processed_array = preprocess_image(original_image, blur_amount, block_size, c_const)
    processed_image = Image.fromarray(processed_array)

    tab1, tab2, tab3 = st.tabs(["🔍 Scan & Analyze", "🖍️ Visual Search", "💾 Export Data"])

    # --- TAB 1: SCANNING ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original", use_container_width=True)
        with col2:
            st.image(processed_image, caption="Computer Vision View", use_container_width=True)

        if st.button("Start OCR Scan", type="primary"):
            with st.spinner("Processing..."):
                extracted_text = extract_text(processed_image)
                
                if extracted_text:
                    st.session_state['text'] = extracted_text
                    st.session_state['data'] = parse_data(extracted_text)
                    st.session_state['proc_img'] = processed_image
                    st.success("Scan Complete!")
                else:
                    st.error("Tesseract failed to read text. Please check the 'Critical Error' at the top.")

        if 'text' in st.session_state:
            data = st.session_state['data']
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Emails", len(data["Emails"]))
            m2.metric("Phones", len(data["Phones"]))
            m3.metric("Dates", len(data["Dates"]))
            m4.metric("Links", len(data["Links"]))
            
            with st.expander("View Raw Extracted Text"):
                st.text_area("", st.session_state['text'], height=200)

    # --- TAB 2: VISUAL SEARCH ---
    with tab2:
        if 'text' in st.session_state:
            st.subheader("Find in Page")
            search_query = st.text_input("Enter word to find:", placeholder="e.g. Total")
            
            if search_query:
                highlighted_img, count = search_and_highlight(
                    st.session_state['proc_img'], 
                    original_image, 
                    search_query
                )
                if count > 0:
                    st.success(f"Found {count} matches.")
                    st.image(highlighted_img, use_container_width=True)
                else:
                    st.warning("Word not found.")
        else:
            st.info("Run the scan first.")

    # --- TAB 3: EXPORT DATA ---
    with tab3:
        if 'text' in st.session_state:
            st.subheader("Download Results")
            data = st.session_state['data']
            structured_items = []
            for category, items in data.items():
                for item in items:
                    structured_items.append({"Category": category, "Value": item})
            
            df = pd.DataFrame(structured_items)
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "scanned_data.csv", "text/csv")
            else:
                st.warning("No structured data found.")
            
            st.download_button("📄 Download Text", st.session_state['text'], "raw_scan.txt", "text/plain")
        else:
            st.info("Run the scan first.")

else:
    st.info("Upload a document to begin.")
