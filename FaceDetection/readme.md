👤 The FaceID Gatekeeper (Week 9)

A Biometric Access Control System built with Python, DeepFace, and Streamlit. This project simulates a secure entry gate that identifies individuals via facial recognition and logs their attendance automatically.

🚀 The Goal

Week 9 of the Computer Vision Roadmap. Moving from general object detection (Week 5-6) to specific biometric identification. The goal was to build a system that can create, store, and match 128-dimensional face embeddings without storing raw images.

🛠️ Tech Stack

DeepFace: A lightweight facial recognition framework (wrapping Google FaceNet, VGG-Face, etc.) for generating embeddings.

Streamlit: For the "Gatekeeper" dashboard UI.

OpenCV & NumPy: For image processing and vector math (Euclidean distance).

Pandas: For managing the attendance ledger (CSV).

Pickle: For serializing the vector database.

✨ Features

Biometric Registration: Extracts 128-D vector embeddings from photos and stores them linked to a name.

The Matcher: Compares live inputs against the database using Euclidean distance (np.linalg.norm).

Gate Logic:

Distance < Threshold: ACCESS GRANTED (Green).

Distance > Threshold: ACCESS DENIED (Red).

Auto-Logging: Automatically records "Time-In" for known users into a CSV, preventing duplicate entries for the same day.

Security Config: Adjustable sensitivity threshold via the sidebar.

📦 Installation

Prerequisites:
You generally do not need C++ compilers for DeepFace (unlike dlib), making it easier to install on Windows.

Install Dependencies:

pip install streamlit deepface tf-keras opencv-python numpy pillow pandas


Run the App:

streamlit run face_id_v2.py


🧠 How it Works

The system does not save photos. It converts faces into numbers.

Elon Musk -> [-0.12, 0.55, 0.91, ...] (128 numbers)

Intruder -> [0.88, -0.22, 0.11, ...]

The system calculates the math distance between these two lists. If they are close, it's the same person.

📜 License

MIT