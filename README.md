# EDI – Explore / Develop / Improve

## Overview
AI-powered sports coaching platform that helps athletes train, improve, and get discovered. Combines AI-powered video analysis, 3D training simulations, and web dashboards.

## Key Features
### Performance Analysis
- Tracks player movements using OpenCV and MediaPipe
- Computes performance metrics: passes, dribbles, speed
- Analyzes textual data from coaches' reports

### Training Simulation
- 3D scenes for correct movement visualization
- Interactive training model (`correct_movements.mp4`)
- Developed in C# (Unity)

### Web Interfaces
- `coach_dribbling.html`: Coach's dashboard
- `home_dribbling.html`: Player monitoring panel

## Technologies
Python (Analysis Module), OpenCV, MediaPipe, NumPy, HTML/CSS/JS (Web Module), C# (Unity)

## How to Run
```bash
pip install opencv-python mediapipe numpy
python app.py
