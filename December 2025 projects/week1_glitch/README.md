Retro Glitch Engine 👾

A Python-based image manipulation tool that applies retro 90s-style "Glitch" and "Vaporwave" aesthetic filters. Built with Streamlit, NumPy, and Pillow.

This tool demonstrates low-level pixel manipulation by treating images as raw NumPy matrices rather than using high-level "magic" AI filters.

Features

Pixel Inspection: View the raw matrix data (RGB arrays) of any uploaded image.

Channel Swap: Manually swaps RGB channels to create surreal, X-ray-like color palettes.

RGB Shift (Chromatic Aberration): Simulates a retro VHS effect by physically shifting the Red and Blue pixel arrays in opposite directions.

Watermarking: Adds a custom, semi-transparent text overlay to the final image.

Tech Stack

Python 3.x

Streamlit: For the web interface.

NumPy: For high-performance matrix operations and pixel shifting.

Pillow (PIL): For image loading, saving, and drawing text.

How to Run Locally

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/retro-glitch-engine.git](https://github.com/YOUR_USERNAME/retro-glitch-engine.git)
cd retro-glitch-engine


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run glitch_engine.py


Example Code Logic

The core logic relies on numpy.roll to shift pixel channels:

# Shifting the Red channel 15 pixels to the left
r_shifted = np.roll(red_channel, -15, axis=1)


License

MIT