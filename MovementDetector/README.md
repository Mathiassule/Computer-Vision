📹 The Invisible Eye (Week 3)

A real-time Motion Detection security system built with Python, OpenCV, and Streamlit.

🚀 The Goal

Week 3 of my Computer Vision Roadmap. Moving from static image processing to Real-Time Video Analysis. This project monitors a webcam feed, calculates background changes, and triggers alarms when movement is detected.

🛠️ Tech Stack

OpenCV (cv2): For accessing the webcam stream and image processing (blur, threshold, contours).

Streamlit: For the live dashboard UI.

NumPy: For matrix subtraction (calculating the "Delta" frame).

✨ Features

Background Subtraction: Automatically calculates the difference between the current frame and a static baseline to identify changes.

Noise Filtering: Uses Gaussian Blur and contour area filtering to ignore small movements (bugs, lighting flickers).

Real-Time Tracking: Draws red bounding boxes around detected intruders.

Security Dashboard: Displays a live feed, the raw motion mask, and a dynamic status alert system using custom CSS for visual alarms.

📸 How It Works (The Logic)

1. The Baseline:
We capture the first frame of the video as the "Empty Room."

2. The Delta:
For every new frame, we calculate:


$$Delta = | CurrentFrame - BaselineFrame |$$

3. The Threshold:
If the pixel difference is > 30, it's motion (White). Otherwise, it's background (Black).

4. Contours:
We find the outline of the white blobs. If the blob area > sensitivity, we trigger the alarm.

📦 How to Run

Install dependencies:

pip install streamlit opencv-python numpy


Run the app locally:

streamlit run security_cam.py


Note: This app requires access to a USB webcam, so it runs best on a local machine rather than a cloud server.

📜 License

MIT