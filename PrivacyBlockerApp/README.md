🛡️ Privacy Blocker AI (Week 2)

An automated privacy tool built with Python and OpenCV that detects faces in images and anonymizes them using Gaussian Blurring or Augmented Reality masking.

🚀 The Goal

Week 2 of my Computer Vision Roadmap. Moving from global pixel manipulation (Week 1) to Object Detection and Region of Interest (ROI) processing.

🛠️ Tech Stack

OpenCV (cv2): For Haar Cascade face detection and image processing.

NumPy: For array slicing and alpha blending math.

Streamlit: For the interactive web interface.

Pillow (PIL): For procedural asset generation (drawing emojis via code).

✨ Features

Face Detection: Uses pre-trained Haar Cascades to locate faces.

Adjustable Sensitivity: Sidebar controls to tune scaleFactor and minNeighbors to reduce false positives.

Privacy Blur: Applies a Gaussian Blur filter to the specific ROI (face area).

Emoji Mask (AR): Uses custom Alpha Blending logic to overlay a generated emoji on top of the face, preserving transparency.

Offline Asset Generation: The emoji overlay is drawn programmatically using PIL, removing external dependencies.

📸 How It Works (The Math)

1. Detection:
The app scans the image matrix for contrast patterns (darker eyes, lighter cheeks) to return coordinates (x, y, w, h).

2. The Blur (ROI Slicing):

# Isolate the face
roi = image[y:y+h, x:x+w]
# Scramble pixels
blurred = cv2.GaussianBlur(roi, (31, 31), 0)
# Inject back
image[y:y+h, x:x+w] = blurred


3. The Mask (Alpha Blending):
To paste the round emoji without black corners, I used this blending formula:
$$ Pixel = (Emoji \times \alpha) + (Background \times (1 - \alpha)) $$

📦 How to Run

Install dependencies:

pip install streamlit opencv-python-headless numpy pillow


Run the app:

streamlit run privacy_blocker.py


📜 License

MIT