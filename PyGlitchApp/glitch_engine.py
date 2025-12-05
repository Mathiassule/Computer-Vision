import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Retro Glitch Engine", page_icon="👾", layout="wide")

st.title("👾 The Retro-Glitch Engine")
st.markdown("""
**Pixel Manipulation Tool**
Upload an image to inspect its matrix data, apply channel shifts, and generate retro glitch aesthetics.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# --- CORE FUNCTIONS ---

def inspect_pixel_data(image):
    """
    Convert image to numpy array and display raw matrix statistics.
    """
    # Convert PIL image to Numpy Array
    img_array = np.array(image)
    
    st.subheader("Pixel Matrix Data 🟢")
    st.write(f"**Dimensions:** {img_array.shape} (Height, Width, Channels)")
    st.write("Raw pixel sample (Top-Left 5x5 slice):")
    st.code(str(img_array[:5, :5, :]), language="python")
    return img_array

def apply_channel_swap(image):
    """
    Swaps RGB channels: Red becomes Blue, etc.
    """
    img_array = np.array(image)
    
    # Handle Alpha channel if present (ignore it for the swap)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Extract channels
    r = img_array[:, :, 0].copy()
    g = img_array[:, :, 1].copy()
    b = img_array[:, :, 2].copy()

    # Swap Red and Blue
    # Stack order becomes: Blue, Green, Red (BGR)
    swapped_array = np.stack([b, g, r], axis=2)
    
    return Image.fromarray(swapped_array)

def apply_rgb_shift(image, shift_amount):
    """
    Simulates chromatic aberration by shifting Red and Blue channels.
    """
    img_array = np.array(image)
    
    # Handle Alpha channel
    has_alpha = False
    if img_array.shape[2] == 4:
        has_alpha = True
        alpha = img_array[:, :, 3]
        img_array = img_array[:, :, :3]

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # Shift Red channel left (negative roll)
    r_shifted = np.roll(r, -shift_amount, axis=1)
    
    # Shift Blue channel right (positive roll)
    b_shifted = np.roll(b, shift_amount, axis=1)
    
    # Green stays centered
    
    # Recombine channels
    glitched_array = np.stack([r_shifted, g, b_shifted], axis=2)
    
    # Restore alpha if it existed
    if has_alpha:
        glitched_array = np.dstack([glitched_array, alpha])

    return Image.fromarray(glitched_array)

def add_watermark(image, text="GLITCH_MODE"):
    """
    Adds a semi-transparent text watermark to the bottom-right corner.
    """
    # Convert to RGBA for transparency handling
    watermarked = image.copy().convert("RGBA")
    
    # Create a transparent overlay layer
    overlay = Image.new("RGBA", watermarked.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Load font (fallback to default if system font missing)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Position text at bottom right with padding
    width, height = watermarked.size
    text_x = width - 150
    text_y = height - 50
    
    # Draw text with 50% opacity (128/255)
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 128), font=font)
    
    # Composite the overlay onto the original
    combined = Image.alpha_composite(watermarked, overlay)
    return combined.convert("RGB")

# --- MAIN APP LOGIC ---

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # Layout: Split screen
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
        
        # Matrix Inspection Tool
        with st.expander("Inspect Pixel Data"):
            inspect_pixel_data(original_image)

    with col2:
        st.subheader("Glitch Controls")
        
        # Effect Selector
        mode = st.radio("Select Filter:", ["None", "Channel Swap", "RGB Shift"])
        
        result_image = original_image
        
        if mode == "Channel Swap":
            result_image = apply_channel_swap(original_image)
            st.caption("Effect applied: RGB Channels Swapped")
            
        elif mode == "RGB Shift":
            shift = st.slider("Shift Intensity (px)", 5, 50, 15)
            result_image = apply_rgb_shift(original_image, shift)
            st.caption(f"Effect applied: Chromatic Aberration ({shift}px)")
            
        st.divider()
        
        # Post-Processing: Watermark
        use_watermark = st.checkbox("Add Watermark")
        
        if use_watermark:
            custom_text = st.text_input("Watermark Text", "RETRO_GLITCH")
            result_image = add_watermark(result_image, custom_text)
            
        # Display Final Result
        st.image(result_image, caption="Processed Output", use_container_width=True)

        # Download Mechanism
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        st.download_button(
            label="💾 Download Result", 
            data=buf.getvalue(), 
            file_name="glitch_art.jpg", 
            mime="image/jpeg"
        )

else:
    st.info("Please upload an image to begin.")