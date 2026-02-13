import streamlit as st
import requests
import io
from PIL import Image
import json
import random
import time
import zipfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Data Factory", page_icon="🏭", layout="wide")

st.title("🏭 AI Synthetic Data Factory")
st.markdown("**Week 11, Day 5: Mass Production Pipeline**")

# --- SIDEBAR: API SETUP ---
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Hugging Face API Token", type="password", help="Get free token from huggingface.co/settings/tokens")

# --- SIDEBAR: PROMPT ENGINEERING ---
st.sidebar.divider()
st.sidebar.subheader("🎛️ Image Parameters")

# 1. Style Presets
style_preset = st.sidebar.selectbox(
    "Style Preset", 
    ["None (Raw)", "Photorealistic", "Cinematic", "Anime/Manga", "Line Art (Black & White)", "3D Model"]
)

# 2. Batch Control
batch_size = st.sidebar.slider("Batch Size", 1, 5, 2, help="Images to generate per run.")

# 3. Seed Control
seed_input = st.sidebar.number_input("Base Seed (-1 for Random)", value=-1, step=1)

# 4. Advanced Params
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
    # Retry logic for model loading
    for _ in range(3):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.content
        elif response.status_code == 503:
            time.sleep(5) # Wait for model to load
            continue
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")
    raise Exception("Model timeout. Try again.")

# --- MAIN APP ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prompting")
    base_prompt = st.text_area("Describe the subject:", height=150, placeholder="A construction worker wearing a safety vest")
    
    generate_btn = st.button(f"Generate Batch ({batch_size})", type="primary", use_container_width=True)
    
    final_prompt = apply_style(base_prompt, style_preset)
    if base_prompt:
        st.info(f"**Final Prompt:** {final_prompt}")

with col2:
    st.subheader("Factory Output")
    
    if generate_btn:
        if not api_key:
            st.error("❌ Please enter your Hugging Face API Token in the sidebar.")
        elif not base_prompt:
            st.warning("Please enter a prompt.")
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            
            # Create a ZIP buffer in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                
                # Grid UI
                image_grid = st.columns(batch_size) if batch_size > 1 else [st.container()]
                progress_bar = st.progress(0)
                
                for i in range(batch_size):
                    # Calculate Seed
                    if seed_input == -1:
                        current_seed = random.randint(0, 2**32 - 1)
                    else:
                        current_seed = int(seed_input) + i
                    
                    payload = {
                        "inputs": final_prompt,
                        "parameters": {
                            "negative_prompt": negative_prompt,
                            "guidance_scale": guidance_scale,
                            "seed": current_seed,
                            "width": 1024,
                            "height": 1024
                        }
                    }
                    
                    # Target Column for UI
                    target_col = image_grid[i] if batch_size > 1 else image_grid[0]
                    
                    with target_col:
                        with st.spinner(f"Gen {i+1}/{batch_size}..."):
                            try:
                                image_bytes = query_model(payload, headers)
                                image = Image.open(io.BytesIO(image_bytes))
                                
                                # Show Image
                                st.image(image, caption=f"Seed: {current_seed}", use_container_width=True)
                                
                                # Add to Zip (Convert Image to Bytes)
                                img_byte_arr = io.BytesIO()
                                image.save(img_byte_arr, format='PNG')
                                filename = f"data_{current_seed}.png"
                                
                                # Write to zip file inside the loop
                                zf.writestr(filename, img_byte_arr.getvalue())
                                
                            except Exception as e:
                                st.error(f"Failed.")
                                st.caption(f"{e}")
                    
                    progress_bar.progress((i + 1) / batch_size)
            
            # --- DOWNLOAD ZIP BUTTON ---
            st.success("Batch Complete!")
            
            st.download_button(
                label="📦 Download Entire Dataset (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="synthetic_dataset.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )