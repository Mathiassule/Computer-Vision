import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Privacy Blocker AI", page_icon="🛡️", layout="wide")

st.title("🛡️ The Privacy Blocker")
st.markdown("""
**Automated Face Anonymization Tool** This tool uses Computer Vision (Haar Cascades) to detect faces and allows you to anonymize them 
via Gaussian Blurring or Augmented Reality masking.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Control Panel")

# Privacy Settings
mode = st.sidebar.radio("Select Mode:", ["View Detections", "Blur Faces", "Emoji Mask"])

# Dynamic Controls based on mode
blur_amount = 0
if mode == "Blur Faces":
    st.sidebar.caption("Adjust the strength of the privacy blur.")
    blur_amount = st.sidebar.slider("Blur Intensity", 1, 99, 31, step=2)

# Detection Settings (Advanced)
with st.sidebar.expander("⚙️ Advanced Detection Settings"):
    st.caption("Adjust these if faces are missed or false positives occur.")
    scale_factor = st.slider("Scale Factor", 1.05, 1.50, 1.1, 0.05)
    min_neighbors = st.slider("Min Neighbors", 1, 15, 6)

# --- ASSET GENERATION ---
@st.cache_data
def get_emoji():
    """
    Generates a high-quality 'Sunglasses' emoji using PIL drawing commands.
    Ensures the app works 100% offline without external dependencies.
    """
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Yellow Face
    draw.ellipse((5, 5, 95, 95), fill="#FFCC4D") 
    
    # Sunglasses
    glasses_color = "#292F33"
    draw.chord((10, 35, 48, 65), start=180, end=0, fill=glasses_color) 
    draw.chord((10, 35, 48, 65), start=0, end=180, fill=glasses_color) 
    draw.chord((52, 35, 90, 65), start=180, end=0, fill=glasses_color)
    draw.chord((52, 35, 90, 65), start=0, end=180, fill=glasses_color)
    draw.rectangle((48, 38, 52, 42), fill=glasses_color)
    
    # Reflections
    draw.ellipse((35, 40, 42, 45), fill="#FFFFFF") 
    draw.ellipse((77, 40, 84, 45), fill="#FFFFFF") 
    
    # Smile
    draw.arc((30, 60, 70, 80), start=20, end=160, fill="#664500", width=4)

    return img

# --- CORE LOGIC ---
def process_image(image, scaleVal, neighVal, mode, blurVal=0):
    # Convert to OpenCV format (RGB -> BGR for processing, but we keep RGB for logic here)
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scaleVal, 
        minNeighbors=neighVal, 
        minSize=(30, 30)
    )
    
    result_img = img_array.copy()
    
    # Pre-load emoji if needed
    emoji_img = None
    if mode == "Emoji Mask":
        emoji_img = get_emoji()

    for (x, y, w, h) in faces:
        if mode == "View Detections":
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
            
        elif mode == "Blur Faces":
            roi = result_img[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (blurVal, blurVal), 0)
            result_img[y:y+h, x:x+w] = blurred_roi
            
        elif mode == "Emoji Mask":
            # Resize emoji to fit face width/height
            emoji_resized = emoji_img.resize((w, h), Image.Resampling.LANCZOS)
            
            # Alpha Blending Logic
            fg_img = np.array(emoji_resized)
            bg_roi = result_img[y:y+h, x:x+w]

            fg_rgb = fg_img[:, :, :3]
            alpha = fg_img[:, :, 3] / 255.0
            alpha_mask = np.dstack((alpha, alpha, alpha))
            
            combined = (fg_rgb * alpha_mask) + (bg_roi * (1.0 - alpha_mask))
            result_img[y:y+h, x:x+w] = combined.astype(np.uint8)
            
    return result_img, len(faces)

# --- MAIN UI ---
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        st.image(uploaded_file, caption="Original", use_container_width=True)

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Run Processing
        result_image, face_count = process_image(image, scale_factor, min_neighbors, mode, blur_amount)
        
        # Display
        st.subheader("Processed Result")
        st.image(result_image, caption=f"Mode: {mode}", use_container_width=True)
        
        # Stats & Download
        if face_count > 0:
            st.success(f"✅ Successfully processed {face_count} face(s).")
            
            # Convert Array back to Image for download
            result_pil = Image.fromarray(result_image)
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG")
            
            st.download_button(
                label="💾 Download Protected Image",
                data=buf.getvalue(),
                file_name="privacy_protected.jpg",
                mime="image/jpeg"
            )
        else:
            st.warning("⚠️ No faces detected. Try adjusting the 'Advanced Detection Settings' in the sidebar.")
    else:
        st.info("👈 Upload an image to get started.")