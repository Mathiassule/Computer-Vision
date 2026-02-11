import streamlit as st
import requests
import io
from PIL import Image
import json
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")

st.title("🎨 AI Synthetic Data Generator")
st.markdown("**Week 11, Day 3: Styles & Reproducibility**")

# --- SIDEBAR: API SETUP ---
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Hugging Face API Token", type="password", help="Get free token from huggingface.co/settings/tokens")

# --- SIDEBAR: PROMPT ENGINEERING ---
st.sidebar.divider()
st.sidebar.subheader("🎛️ Image Parameters")

# 1. Style Presets (New for Day 3)
style_preset = st.sidebar.selectbox(
    "Style Preset", 
    ["None (Raw)", "Photorealistic", "Cinematic", "Anime/Manga", "Line Art (Black & White)", "3D Model"]
)

# 2. Seed Control (New for Day 3)
# -1 means Random. Any other number locks the generation.
seed_input = st.sidebar.number_input("Seed (-1 for Random)", value=-1, step=1, help="Set a specific number to reproduce the exact same image again.")

# 3. Advanced Params
negative_prompt = st.sidebar.text_area("Negative Prompt:", value="blurry, low quality, distorted, ugly, bad anatomy, watermark, text", height=100)
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5)

# --- CONSTANTS ---
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

# --- CORE LOGIC ---
def apply_style(prompt, style):
    if style == "Photorealistic":
        return f"{prompt}, photorealistic, 8k, highly detailed, raw photo, dslr"
    elif style == "Cinematic":
        return f"{prompt}, cinematic lighting, movie scene, dramatic, shallow depth of field, anamorphic lens"
    elif style == "Anime/Manga":
        return f"{prompt}, anime style, studio ghibli, vibrant colors, cel shaded"
    elif style == "Line Art (Black & White)":
        return f"{prompt}, line art, black and white, sketch, ink drawing, technical drawing, white background"
    elif style == "3D Model":
        return f"{prompt}, 3d render, unreal engine 5, octane render, isometric"
    return prompt

def query_model(payload, headers):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    return response.content

# --- MAIN APP ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Prompting")
    base_prompt = st.text_area("Describe the subject:", height=150, placeholder="A construction worker wearing a safety vest")
    
    generate_btn = st.button("Generate Synthetic Data", type="primary", use_container_width=True)
    
    # Show active parameters
    final_prompt = apply_style(base_prompt, style_preset)
    if base_prompt:
        st.info(f"**Final Prompt:** {final_prompt}")

with col2:
    st.subheader("Output")
    
    if generate_btn:
        if not api_key:
            st.error("❌ Please enter your Hugging Face API Token in the sidebar.")
        elif not base_prompt:
            st.warning("Please enter a prompt.")
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Determine Seed
            if seed_input == -1:
                active_seed = random.randint(0, 2**32 - 1)
            else:
                active_seed = int(seed_input)
            
            # Payload with Seed
            payload = {
                "inputs": final_prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "guidance_scale": guidance_scale,
                    "seed": active_seed,
                    # SDXL Native Resolution
                    "width": 1024,
                    "height": 1024
                }
            }
            
            with st.spinner(f"Generating with Seed: {active_seed}..."):
                try:
                    image_bytes = query_model(payload, headers)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    st.image(image, caption=f"Style: {style_preset} | Seed: {active_seed}", use_container_width=True)
                    
                    # Download
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    filename = f"synthetic_{style_preset.lower().replace(' ', '_')}_{active_seed}.png"
                    
                    st.download_button(
                        label="💾 Download Image",
                        data=buf.getvalue(),
                        file_name=filename,
                        mime="image/png"
                    )
                    
                    st.success(f"Generated successfully! Seed used: {active_seed}")
                    
                except Exception as e:
                    st.error("Generation Failed.")
                    st.warning(f"Details: {e}")
                    if "503" in str(e):
                        st.info("💡 Model is loading (Cold Start). Please wait 30s and try again.")
                    elif "403" in str(e):
                        st.info("💡 403 Error: Check your API Token permissions. Ensure 'Inference' is enabled.")