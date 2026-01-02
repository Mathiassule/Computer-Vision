🛡️ The AI Security Guard (Week 5)

A real-time Object Detection and Security Dashboard built with YOLOv8, OpenCV, and Streamlit.

🚀 The Goal

Week 5 of my Computer Vision Roadmap. The objective was to move from simple "Motion Detection" (Pixel changes) to intelligent "Object Detection" (Understanding what the object is).

🛠️ Tech Stack

YOLOv8 (Ultralytics): The state-of-the-art Neural Network for real-time object detection.

OpenCV: For video stream handling and drawing visual alarms.

Streamlit: For the interactive dashboard and logic controls.

Pandas: For logging security incidents to a structured dataset.

✨ Features

Real-Time Detection: Identifies 80+ objects (People, Cars, Phones, Weapons, etc.) instantly via webcam.

Class Filtering: A dynamic filter system to ignore "noise" and focus only on specific targets (e.g., "Only detect Cell Phones").

Logic Layer: An automated rule engine. You set the rules (e.g., "Max 0 People Allowed"), and the AI triggers if the rule is broken.

Incident Logging: Automatically records timestamps and details of every security breach into a downloadable CSV log.

📦 Installation

Install Dependencies:

pip install streamlit ultralytics opencv-python pandas


Run the App:

streamlit run yolo_app.py


Note: The first run will automatically download the yolov8n.pt model (~6MB).

📜 License

MIT