🏋️‍♂️ The AI Personal Trainer (Week 8)

A real-time AI Fitness Coach built with MediaPipe Pose and Python. It automatically counts reps, corrects form, and supports multiple exercises.

🚀 The Goal

Week 8 of my Computer Vision Roadmap. Moving from Object Detection to Skeletal Pose Estimation. The goal was to build a system that understands human movement geometry.

🛠️ Tech Stack

MediaPipe Pose: Tracks 33 3D body landmarks (Shoulders, Knees, Hips, etc.) with high fidelity.

OpenCV: For video processing and drawing the workout dashboard.

Streamlit: For the web interface and exercise selection.

NumPy: For trigonometric calculations (Arctangent for joint angles).

✨ Features

Multi-Exercise Support:

Bicep Curls: Tracks Elbow Angle.

Squats: Tracks Knee Angle.

Pushups: Tracks Elbow Angle + Body Alignment.

Rep Counting: Implements a State Machine (Up/Down logic) to count only full, valid reps.

Form Correction:

Detects leaning/slouching during curls.

Detects chest collapsing during squats.

Detects sagging hips during pushups.

Visual Feedback: Real-time overlay showing Reps, Stage (Up/Down), and Status (Good/Bad Form).

📦 Installation

Install Dependencies:

pip install streamlit mediapipe opencv-python numpy pillow


Run the App:

streamlit run ai_trainer.py


Note: Requires a webcam for live tracking.

📜 License

MIT