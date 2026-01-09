🏗️ The PPE Detector (Week 6)

A Custom Object Detection system trained to recognize Construction Safety Gear (Hardhats, Vests) using YOLOv8.

🚀 The Goal

Week 6 of my Computer Vision Roadmap. The objective was to move beyond pre-trained models and master Transfer Learning. I curated a custom dataset, trained a YOLOv8 model on a GPU, and deployed it to a Compliance Dashboard.

🛠️ Tech Stack

YOLOv8 (Ultralytics): Custom-trained on Google Colab (Tesla T4 GPU).

Roboflow: Used for dataset sourcing and management.

Streamlit: For the Safety Dashboard UI.

OpenCV: For real-time video frame processing and annotation.

✨ Features

Custom Classes: Detects Hardhat, No-Hardhat, Vest, and Person.

Compliance Logic: Maps detections to "Safe" (Green) or "Unsafe" (Red) status.

Video Analysis: Processes CCTV footage frame-by-frame to track worker safety in motion.

Reporting: Calculates real-time Site Compliance Scores (%).

🧠 Training Metrics

Dataset: ~500 Images (Construction Site Safety)

Epochs: 25

mAP50: > 0.85 (Achieved on validation set)

📦 How to Run

Install Dependencies:

pip install streamlit ultralytics opencv-python numpy


Add Your Model:
Place your trained best.pt file in the project root.

Run the App:

streamlit run ppe_dashboard.py


📜 License

MIT