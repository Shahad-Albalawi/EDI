# EDI – Explore / Develop / Improve

## Overview 📌
**EDI** is an AI-powered sports coaching platform that helps athletes train, improve, and get discovered. The system combines:
- AI-powered video analysis 📹 to track player movements
- 3D training simulations 🎮 to visualize correct techniques
- Web dashboards 🌐 for coaches and players to monitor performance

---

## Key Features ✨

### Performance Analysis 🖥️
- Tracks player movements precisely using **OpenCV** and **MediaPipe**
- Computes player performance metrics: passes, dribbles, speed
- Analyzes textual data from coaches' reports

### Training Simulation 🎮
- 3D scenes for correct movement simulations
- Interactive training model (`correct_movements.mp4`)
- Unity engine for movement visualization
- Developed in **C#** (`Assembly-CSharp.csproj`, `DSS.sln`)

### Web Interfaces 🌐
- `coach_dribbling.html`: Coach's dashboard
- `home_dribbling.html`: Player monitoring panel

---

## Technologies Used ⚙️

### Analysis Module (Python)
- OpenCV 🖥️ – Video processing
- MediaPipe 📹 – Movement tracking
- NumPy 📊 – Data processing

### Web Module (HTML/JS)
- HTML/CSS 🌐 – Structure and styling
- JavaScript 💻 – Interactivity and dynamic behaviors

---

## How to Run 🚀

### Analysis Module
```bash
pip install opencv-python mediapipe numpy
python app.py
