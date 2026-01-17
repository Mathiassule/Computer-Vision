✋ The Touchless Interface (Week 7)

A real-time Hand Tracking & Gesture Control system built with MediaPipe, Python, and Streamlit. This project allows users to control their computer mouse and volume using hand gestures.

🚀 The Goal

Week 7 of my Computer Vision Roadmap. Moving from Object Detection (YOLO) to Skeletal Tracking (MediaPipe). The goal was to build a Human-Computer Interface (HCI) that requires zero physical contact.

🛠️ Tech Stack

MediaPipe: Google's framework for high-fidelity hand tracking (21 landmarks).

OpenCV: For video processing and drawing visual feedback.

Streamlit: For the web interface.

PyAutoGUI: For programmatically controlling the OS mouse and keyboard.

✨ Features

Finger Counter: Uses coordinate geometry to count open fingers (Binary Logic).

Volume Knob: Maps the Euclidean distance between Thumb and Index finger to a percentage scale (0-100%).

Gesture Recognition: Identifies specific patterns (Peace Sign, Fist, Rock On) by analyzing finger states.

Virtual Mouse: Maps the Index finger position to screen coordinates with smoothing algorithms for fluid control. Clicking is triggered by bringing the Index and Middle fingers together.

📦 Installation

Install Dependencies:

pip install streamlit mediapipe opencv-python numpy pillow pyautogui


Run the App:

streamlit run hand_tracker.py


Note: The "Virtual Mouse" feature works best locally on Windows/Mac/Linux. It will not work on Streamlit Cloud as cloud servers do not have a monitor/cursor.

📜 License

MIT