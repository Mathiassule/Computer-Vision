import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="PPE Compliance Logic", page_icon="👷", layout="wide")

st.title("👷 PPE Compliance Monitor")
st.markdown("**Week 6, Day 4: Visualizing Safety Rules**")

# --- LOAD MODEL ---
model_path = 'best.pt'

try:
    model = YOLO(model_path)
    # Get the list of class names the model knows (e.g. ['helmet', 'no-helmet', 'vest'])
    class_names = list(model.names.values())
except Exception as e:
    st.error(f"❌ Could not load 'best.pt'. Make sure it is in the folder.")
    st.stop()

# --- SIDEBAR: SAFETY CONFIGURATOR ---
st.sidebar.header("⚙️ Safety Rules")
st.sidebar.info("Map your model's classes to safety status.")

# Let the user decide what counts as "Safe" vs "Unsafe" based on their specific dataset
safe_classes = st.sidebar.multiselect(
    "✅ Select SAFE Classes (Green)", 
    options=class_names,
    default=[name for name in class_names if 'helmet' in name.lower() or 'vest' in name.lower()]
)

unsafe_classes = st.sidebar.multiselect(
    "❌ Select UNSAFE Classes (Red)", 
    options=class_names,
    default=[name for name in class_names if 'no' in name.lower() or 'head' in name.lower()]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# --- HELPER: COLOR LOGIC ---
def get_color(class_name):
    if class_name in safe_classes:
        return (0, 255, 0) # Green
    elif class_name in unsafe_classes:
        return (255, 0, 0) # Red
    else:
        return (128, 128, 128) # Grey (Neutral/Unknown)

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload Site Photo", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    # Load Image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with col1:
        st.subheader("Input Feed")
        st.image(image, use_container_width=True)

    if st.button("Check Compliance", type="primary"):
        # Run Inference
        results = model(image, conf=confidence)
        
        # We need to draw our own boxes to control the colors dynamically!
        # YOLO's default .plot() handles colors automatically, but we want Custom Logic.
        
        # Create a copy to draw on (OpenCV uses BGR)
        annotated_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        boxes = results[0].boxes
        compliance_score = {"Safe": 0, "Unsafe": 0}
        
        for box in boxes:
            # 1. Get Class Name
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            # 2. Get Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 3. Determine Color
            color = get_color(cls_name)
            
            # 4. Draw Box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            
            # 5. Draw Label Background
            label = f"{cls_name} {box.conf[0]:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 6. Update Stats
            if cls_name in safe_classes:
                compliance_score["Safe"] += 1
            elif cls_name in unsafe_classes:
                compliance_score["Unsafe"] += 1

        # Convert back to RGB for Streamlit
        final_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("Compliance Analysis")
            st.image(final_img, use_container_width=True)
            
        # --- REPORT CARD ---
        st.divider()
        st.write("### 📋 Site Report")
        m1, m2, m3 = st.columns(3)
        m1.metric("Safe Workers", compliance_score["Safe"])
        m2.metric("Safety Violations", compliance_score["Unsafe"], delta_color="inverse")
        
        total = compliance_score["Safe"] + compliance_score["Unsafe"]
        if total > 0:
            rate = round((compliance_score["Safe"] / total) * 100, 1)
            m3.metric("Compliance Rate", f"{rate}%")
            
            if rate < 100:
                st.error("🚨 SAFETY VIOLATION DETECTED: Compliance < 100%")
            else:
                st.success("✅ Site is 100% Compliant")