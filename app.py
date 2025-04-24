from flask import Flask, request, jsonify, render_template, send_from_directory, Response, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import openai
import os
import time
import threading
import uuid
import json
from collections import deque
import sys
import traceback
import logging
import azure.cognitiveservices.speech as speechsdk
import requests
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("dribbling_analysis.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', template_folder='.')
app.secret_key = 'football_analysis_session_key'  # For session management

# Create directories for audio files
os.makedirs('audio/reports', exist_ok=True)
os.makedirs('audio/alerts', exist_ok=True)

# Store analysis results
analysis_storage = {}

# API Key setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
SPEECH_KEY = os.environ.get('SPEECH_KEY', '4JHJjj9COPOqAP9g61cIr6dNgnAF6VSakY2usJpykAD5cNcJWgSWJQQJ99BDACYeBjFXJ3w3AAAYACOGrYMb')
SPEECH_REGION = os.environ.get('SPEECH_REGION', 'eastus')

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Configure Azure Speech Services
speech_config = None
try:
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = "en-US-GuyNeural"
    speech_services_available = True
    logger.info("Azure Speech Services configured successfully")
except Exception as e:
    logger.warning(f"Could not configure Azure Speech Services: {e}")
    speech_services_available = False

# MediaPipe setup with improved configuration
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Improved drawing specs for better visualization
pose_connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
pose_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2)

# Global variables for video streaming
output_frame = None
lock = threading.Lock()
analysis_results = {"dribbles": 0, "Feinting": 0, "warnings": "", "tips": "", "time": 0, "bad_dribbles": 0, "progress": 0}
is_analyzing = False

# ======= Text-to-Speech Functions =======

def text_to_speech_azure(text, output_file=None):
    """Convert text to speech using Azure Speech Services"""
    if not speech_services_available:
        logger.warning("Azure Speech Services not available")
        return None
    
    try:
        # Create speech synthesizer
        if output_file:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        else:
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        
        # Synthesize speech
        result = speech_synthesizer.speak_text_async(text).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Speech synthesis successful")
            if output_file:
                return output_file
            else:
                return result.audio_data
        else:
            logger.error(f"Speech synthesis failed: {result.reason}")
            return None
    except Exception as e:
        logger.error(f"Error in Azure text-to-speech: {e}")
        return None

def text_to_speech_elevenlabs(text, output_file=None):
    """Convert text to speech using ElevenLabs API"""
    try:
        # ElevenLabs API Key
        api_key = SPEECH_KEY  # Reusing the provided speech key
        
        if not api_key or api_key == "":
            logger.warning("ElevenLabs API key not provided")
            return None
        
        # Default voice ID (Jason)
        voice_id = "TxGEqnHWrfWFTfGW9XjX"
        
        # API endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        
        # Request headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        # Request body
        body = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        # Send request
        response = requests.post(url, json=body, headers=headers, stream=True)
        
        if response.status_code == 200:
            # Save the audio to a file if requested
            if output_file:
                with open(output_file, "wb") as out:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            out.write(chunk)
                return output_file
            
            # Otherwise, return the audio content directly
            audio_content = response.content
            return audio_content
        else:
            logger.error(f"Error from ElevenLabs API: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error in ElevenLabs text-to-speech: {e}")
        return None

def text_to_speech(text, output_file=None, voice_name=None):
    """Unified interface for text-to-speech services"""
    # Try ElevenLabs first (better quality)
    audio_content = text_to_speech_elevenlabs(text, output_file)
    
    # Fallback to Azure if ElevenLabs fails
    if audio_content is None:
        # If voice name is provided, try to set it in Azure
        if voice_name and speech_config:
            speech_config.speech_synthesis_voice_name = voice_name
        
        audio_content = text_to_speech_azure(text, output_file)
    
    # Log result
    if audio_content:
        logger.info(f"Text-to-speech successful, length: {len(audio_content) if isinstance(audio_content, bytes) else 'file'}")
    else:
        logger.warning(f"Text-to-speech failed for text: {text[:50]}...")
    
    return audio_content

# ======= Routes for Text-to-Speech =======

@app.route('/api/text_to_speech', methods=['POST'])
def api_text_to_speech():
    """API endpoint for text-to-speech conversion"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        voice_name = data.get('voice', None)
        analysis_id = data.get('analysis_id', '')
        
        # Generate unique filename based on a hash of the text + timestamp
        timestamp = int(time.time())
        filename = f"audio/reports/speech_{timestamp}_{hash(text) % 10000}.mp3"
        
        # Convert text to speech
        result = text_to_speech(text, filename, voice_name)
        
        if result:
            return jsonify({
                "success": True,
                "audio_url": f"/{filename}"
            })
        else:
            return jsonify({"error": "Failed to generate speech"}), 500
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_report_audio', methods=['POST'])
def generate_report_audio():
    """Generate audio for analysis report"""
    try:
        data = request.json
        if not data or 'summary' not in data:
            return jsonify({"error": "No summary provided"}), 400
        
        analysis_id = data.get('analysis_id', str(uuid.uuid4()))
        summary = data['summary']
        
        # Generate filename
        filename = f"audio/reports/report_{analysis_id}.mp3"
        
        # Convert text to speech
        result = text_to_speech(summary, filename)
        
        if result:
            return jsonify({
                "success": True,
                "audio_url": f"/{filename}"
            })
        else:
            return jsonify({"error": "Failed to generate report audio"}), 500
    except Exception as e:
        logger.error(f"Error generating report audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts_stream')
def alerts_stream():
    """Stream for sending real-time alerts"""
    def generate():
        while True:
            try:
                # Send heartbeat every second to keep connection alive
                time.sleep(1)
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except Exception as e:
                logger.error(f"Error in alerts stream: {e}")
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/generate_alert_sounds', methods=['POST'])
def generate_alert_sounds():
    """Generate alert sounds for bad dribbling and feinting"""
    try:
        # Ensure directory exists
        os.makedirs('audio/alerts', exist_ok=True)
        
        # Generate bad dribble alert sound
        bad_dribble_text = "Keep the ball closer to your feet! Try to maintain the ball within 60 centimeters."
        bad_dribble_file = "audio/alerts/bad_dribble.mp3"
        
        # Always regenerate the files to ensure they work
        logger.info(f"Generating bad dribble alert sound to {bad_dribble_file}")
        dribble_result = text_to_speech(bad_dribble_text, bad_dribble_file)
        logger.info(f"Bad dribble sound generation result: {dribble_result}")
        
        # Generate bad feinting alert sound
        bad_feint_text = "Improve your feinting by keeping the ball closer to your feet. Try quicker foot movements."
        bad_feint_file = "audio/alerts/bad_feint.mp3"
        
        logger.info(f"Generating bad feint alert sound to {bad_feint_file}")
        feint_result = text_to_speech(bad_feint_text, bad_feint_file)
        logger.info(f"Bad feint sound generation result: {feint_result}")
        
        # Check if files exist
        dribble_exists = os.path.exists(bad_dribble_file)
        feint_exists = os.path.exists(bad_feint_file)
        
        logger.info(f"Alert sound files exist? Dribble: {dribble_exists}, Feint: {feint_exists}")
        
        return jsonify({
            "success": dribble_exists and feint_exists,
            "dribble_file": bad_dribble_file if dribble_exists else None,
            "feint_file": bad_feint_file if feint_exists else None,
            "message": "Alert sounds generated successfully" if (dribble_exists and feint_exists) else "Some alert sounds failed to generate"
        })
    except Exception as e:
        logger.error(f"Error generating alert sounds: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/play_alert/<alert_type>', methods=['POST'])
def play_alert(alert_type):
    """API endpoint to trigger playing an alert sound"""
    try:
        if alert_type not in ['bad_dribble', 'bad_feint']:
            return jsonify({"error": "Invalid alert type"}), 400
        
        sound_file = f"audio/alerts/{alert_type}.mp3"
        
        # Check if file exists
        if not os.path.exists(sound_file):
            logger.warning(f"Alert sound file {sound_file} does not exist. Generating it...")
            generate_alert_sounds()
        
        # Check again after generating
        if not os.path.exists(sound_file):
            return jsonify({"error": f"Alert sound file {sound_file} could not be generated"}), 500
        
        return jsonify({
            "success": True,
            "message": f"Alert {alert_type} triggered",
            "sound_file": sound_file
        })
    except Exception as e:
        logger.error(f"Error playing alert: {e}")
        return jsonify({"error": str(e)}), 500

# Define a class for movement tracking with Kalman filtering for smoother predictions
class MovementTracker:
    def __init__(self, smooth_factor=0.8):
        self.prev_positions = []
        self.smooth_factor = smooth_factor
        self.history = deque(maxlen=30)  # Keep last 30 positions for trajectory analysis
        self.velocity = [0, 0]
        self.last_update_time = None
    
    def update(self, position):
        if position is None:
            return None
        
        try:
            current_time = time.time()
            
            # Add to history
            self.history.append((position, current_time))
            
            # Calculate velocity if possible
            if self.last_update_time is not None and len(self.prev_positions) > 0:
                dt = current_time - self.last_update_time
                if dt > 0:
                    dx = position[0] - self.prev_positions[-1][0]
                    dy = position[1] - self.prev_positions[-1][1]
                    self.velocity = [dx/dt, dy/dt]
            
            # Apply smoothing
            smoothed_position = position
            if self.prev_positions:
                smoothed_position = (
                    int(self.smooth_factor * position[0] + (1 - self.smooth_factor) * self.prev_positions[-1][0]),
                    int(self.smooth_factor * position[1] + (1 - self.smooth_factor) * self.prev_positions[-1][1])
                )
            
            self.prev_positions.append(smoothed_position)
            if len(self.prev_positions) > 10:
                self.prev_positions.pop(0)
                
            self.last_update_time = current_time
            return smoothed_position
        except Exception as e:
            logger.error(f"Error in movement tracker update: {e}")
            return position  # Return original position if smoothing fails
    
    def get_speed(self):
        try:
            return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        except Exception as e:
            logger.error(f"Error calculating speed: {e}")
            return 0
    
    def get_acceleration(self, window=5):
        """Calculate acceleration based on velocity changes"""
        try:
            if len(self.history) < window + 1:
                return 0
            
            recent = list(self.history)[-window:]
            if len(recent) < 2:
                return 0
            
            # Get first and last velocity measurement
            p1, t1 = recent[0]
            p2, t2 = recent[-1]
            
            if t2 == t1:
                return 0
                
            v1 = np.array([0, 0])  # Initial velocity
            v2 = np.array([(p2[0] - p1[0])/(t2 - t1), (p2[1] - p1[1])/(t2 - t1)])  # Final velocity
            
            # Acceleration = change in velocity / change in time
            acceleration = np.linalg.norm(v2 - v1) / (t2 - t1)
            return acceleration
        except Exception as e:
            logger.error(f"Error calculating acceleration: {e}")
            return 0
    
    def get_movement_direction(self):
        """Get the current movement direction as an angle in degrees"""
        try:
            if abs(self.velocity[0]) < 0.1 and abs(self.velocity[1]) < 0.1:
                return None  # No significant movement
            
            angle_rad = np.arctan2(self.velocity[1], self.velocity[0])
            angle_deg = np.degrees(angle_rad)
            return angle_deg
        except Exception as e:
            logger.error(f"Error calculating movement direction: {e}")
            return None
    
    def get_trajectory(self):
        """Return a list of position points for drawing trajectory"""
        try:
            return [pos for pos, _ in self.history]
        except Exception as e:
            logger.error(f"Error getting trajectory: {e}")
            return []

# Create trackers
ball_tracker = MovementTracker(smooth_factor=0.7)  # Less smoothing for the ball
left_foot_tracker = MovementTracker(smooth_factor=0.85)
right_foot_tracker = MovementTracker(smooth_factor=0.85)

@app.route('/')
def index():
    try:
        return render_template('home_dribbling.html')
    except Exception as e:
        logger.error(f"Error rendering home template: {e}")
        return "Error loading home page. Check if home_dribbling.html exists."

@app.route('/<path:path>')
def serve_html(path):
    try:
        if path.endswith('.html'):
            return render_template(path)
        return send_from_directory('.', path)
    except Exception as e:
        logger.error(f"Error serving {path}: {e}")
        return f"Error loading {path}. File may not exist."

# Route for serving static files like audio files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def generate_frames():
    global output_frame, lock, is_analyzing
    
    while True:
        try:
            with lock:
                if output_frame is None or not is_analyzing:
                    # If no frame is being processed, send a placeholder
                    if not is_analyzing:
                        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 255
                        cv2.putText(placeholder, "Waiting for dribbling video analysis to start...", 
                                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        _, buffer = cv2.imencode('.jpg', placeholder)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        time.sleep(0.5)
                        continue
                    continue
                
                # Encode the frame to JPEG
                _, buffer = cv2.imencode('.jpg', output_frame)
                frame = buffer.tobytes()
            
            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            # Create error frame
            error_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(error_frame, "Error processing video frame", 
                       (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0)  # Longer delay on error

@app.route('/video_feed')
def video_feed():
    """Route for streaming the analyzed video."""
    try:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed: {e}")
        return "Video feed error", 500

@app.route('/analysis_status')
def analysis_status():
    """Route to get the current analysis results."""
    global analysis_results, is_analyzing
    try:
        has_new_warning = False
        if 'last_warning' not in session:
            session['last_warning'] = ""
            session['last_warning_time'] = 0
        
        # Check if we have a new warning that we haven't shown recently
        current_time = time.time()
        if (analysis_results.get("warnings", "") and 
            analysis_results.get("warnings", "") != session.get('last_warning', "") and
            current_time - session.get('last_warning_time', 0) > 3):  # Only trigger a new warning every 3 seconds
            
            has_new_warning = True
            session['last_warning'] = analysis_results.get("warnings", "")
            session['last_warning_time'] = current_time
        
        return jsonify({
            "is_analyzing": is_analyzing,
            "results": analysis_results,
            "new_warning": has_new_warning
        })
    except Exception as e:
        logger.error(f"Error in analysis_status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/play_alert', methods=['POST'])
def old_play_alert():
    """API endpoint to notify that we should play an alert (legacy)"""
    try:
        alert_type = request.json.get('alert_type', 'bad_dribble')
        # Forward to the new API
        return play_alert(alert_type)
    except Exception as e:
        logger.error(f"Error in play_alert: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_analysis_thumbnail', methods=['POST'])
def save_analysis_thumbnail():
    """Save a thumbnail from the analyzed video for coach view"""
    global output_frame, lock
    
    try:
        analysis_id = request.form.get('analysis_id')
        if not analysis_id or analysis_id not in analysis_storage:
            return jsonify({"error": "Invalid analysis ID"}), 400
        
        with lock:
            if output_frame is not None:
                # Save thumbnail
                thumbnail_path = f"analysis_{analysis_id}_thumbnail.jpg"
                cv2.imwrite(thumbnail_path, output_frame)
                analysis_storage[analysis_id]["thumbnail"] = thumbnail_path
                
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving thumbnail: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_analysis/<analysis_id>')
def get_analysis(analysis_id):
    """Get specific analysis results"""
    try:
        if analysis_id not in analysis_storage:
            return jsonify({"error": "Analysis not found"}), 404
            
        return jsonify(analysis_storage[analysis_id])
    except Exception as e:
        logger.error(f"Error getting analysis {analysis_id}: {e}")
        return jsonify({"error": str(e)}), 500

def detect_ball(frame, min_radius=10, max_radius=30):
    """Improved ball detection with multi-technique approach"""
    try:
        # Safety check for frame
        if frame is None or frame.size == 0:
            return None, 0, frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Try to detect circular objects (balls) using Hough Circle Transform
        try:
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1.2,            # Resolution ratio
                minDist=50,        # Min distance between circles
                param1=100,        # Higher threshold for Canny edge detection
                param2=30,         # Threshold for circle detection
                minRadius=min_radius,
                maxRadius=max_radius
            )
        except cv2.error as cv_err:
            logger.warning(f"OpenCV error in HoughCircles: {cv_err}")
            return None, 0, frame
        
        ball_position = None
        confidence = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Find the most likely ball (strongest circle)
            best_circle = None
            best_score = 0
            
            for circle in circles[0, :]:
                # Extract the region of the potential ball
                x, y, r = circle
                
                # Safety check for boundaries
                if x - r < 0 or y - r < 0 or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
                    continue
                    
                # Calculate confidence score based on circularity and color
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Check for typical ball colors (white/black areas)
                ball_region = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Calculate average color and standard deviation
                if np.sum(mask) > 0:  # Avoid division by zero
                    mean_color = cv2.mean(ball_region, mask=mask)
                    
                    # For a typical ball, we expect high variance in colors (black and white patterns)
                    # And generally brighter than surroundings
                    brightness = (mean_color[0] + mean_color[1] + mean_color[2])/3
                    
                    # Simple scoring system
                    score = brightness * r  # Favor larger, brighter circles
                    
                    if score > best_score:
                        best_score = score
                        best_circle = circle
                        confidence = min(1.0, score / 5000)  # Normalize confidence
            
            if best_circle is not None:
                x, y, r = best_circle
                ball_position = (x, y)
                
                # Draw the ball with confidence indicator
                color = (0, int(255 * confidence), 0)  # More green = higher confidence
                cv2.circle(frame, ball_position, r, color, 2)
                cv2.circle(frame, ball_position, 2, (0, 0, 255), 3)
                
                # Show confidence
                cv2.putText(frame, f"Ball {confidence:.2f}", (x - r, y - r - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return ball_position, confidence, frame
    except Exception as e:
        logger.error(f"Error in ball detection: {e}")
        return None, 0, frame

def calculate_angle(a, b, c):
    """Calculate angle between three points (used for joint angles)"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Handle potential errors with invalid coordinates
        if np.array_equal(a, b) or np.array_equal(b, c):
            return 0
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle
    except Exception as e:
        logger.error(f"Error calculating angle: {e}")
        return 0

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    try:
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return 0

def analyze_video_thread(video_path, analysis_id):
    global output_frame, lock, analysis_results, is_analyzing, analysis_storage
    global ball_tracker, left_foot_tracker, right_foot_tracker
    
    # Reset trackers for this new analysis
    ball_tracker = MovementTracker(smooth_factor=0.7)
    left_foot_tracker = MovementTracker(smooth_factor=0.85)
    right_foot_tracker = MovementTracker(smooth_factor=0.85)
    
    try:
        logger.info(f"Starting dribbling analysis of video: {video_path}")
        is_analyzing = True
        
        # Initialize progress tracking
        with lock:
            analysis_results["progress"] = 5
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["progress"] = 5
        
        # Generate alert sounds in advance
        try:
            requests.post('http://localhost:5000/api/generate_alert_sounds')
        except Exception as e:
            logger.warning(f"Could not pre-generate alert sounds: {e}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            with lock:
                analysis_results = {"error": "Video file not found.", "progress": 0}
                if analysis_id in analysis_storage:
                    analysis_storage[analysis_id]["error"] = "Video file not found."
            is_analyzing = False
            return
        
        # Update progress
        with lock:
            analysis_results["progress"] = 10
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["progress"] = 10
        
        # Try to open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            with lock:
                analysis_results = {"error": "Could not open the video file.", "progress": 0}
                if analysis_id in analysis_storage:
                    analysis_storage[analysis_id]["error"] = "Could not open the video file."
            is_analyzing = False
            return
        
        # Check video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
            logger.error(f"Invalid video properties: fps={fps}, frames={total_frames}, size={width}x{height}")
            with lock:
                analysis_results = {"error": "Invalid video format.", "progress": 0}
                if analysis_id in analysis_storage:
                    analysis_storage[analysis_id]["error"] = "Invalid video format."
            is_analyzing = False
            cap.release()
            return
        
        # Update progress
        with lock:
            analysis_results["progress"] = 15
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["progress"] = 15
        
        # Football metrics - FOCUS ONLY ON DRIBBLING
        dribbles_counter = 0
        bad_dribbles_counter = 0  # New counter for bad dribbles
        ball_control_score = 100
        dribble_quality_sum = 0  # For tracking average quality
        processed_frames = 0
        
        # Bad dribble reasons tracking
        bad_dribble_reasons = {}  # To track reasons for bad dribbles
        
        # Skill-specific metrics for Feinting (rapid dribbling)
        Feinting_counter = 0
        Feinting_quality = 0  # Initialize the missing variable
        last_Feinting_time = 0
        Feinting_sequence = []
        
        # Advanced metrics for dribbling only
        movements_data = {
            "dribbles": [],
            "Feinting": []
        }
        
        # Game state tracking
        start_time = time.time()
        gpt_tips_for_overlay = ""
        warnings_list = []
        performance_score = 100
        
        # Event cooldowns to prevent duplicate detections
        event_cooldowns = {
            "dribble": 0,
            "Feinting": 0,
            "warning": 0,  # Add cooldown for warnings
            "bad_dribble": 0,
            "bad_feint": 0
        }
        
        # Update progress before starting pose detection
        with lock:
            analysis_results["progress"] = 20
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["progress"] = 20
        
        # Use a less complex model for better performance and stability
        model_complexity = 1  # Medium complexity (0=light, 1=medium, 2=heavy)
        
        # Set up the improved pose detector with balanced confidence
        with mp_pose.Pose(
            min_detection_confidence=0.6,  # Lower threshold to avoid missing poses
            min_tracking_confidence=0.6,
            model_complexity=model_complexity
        ) as pose:
            frame_count = 0
            skip_frames = 2  # Process every nth frame for performance
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                frame_count += 1
                processed_frames += 1
                
                # Calculate progress percentage (from 20% to 90%)
                progress_percent = 20 + min(70, int((processed_frames / total_frames) * 70))
                
                # Update progress every 10 frames
                if processed_frames % 10 == 0:
                    with lock:
                        analysis_results["progress"] = progress_percent
                        if analysis_id in analysis_storage:
                            analysis_storage[analysis_id]["progress"] = progress_percent
                
                # Process only every nth frame for efficiency
                if frame_count % skip_frames != 0:
                    continue
                
                # Initialize warning variable
                warning = ""
                
                try:
                    # Resize for faster processing and consistent display
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Process frame for pose detection
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with timeout protection
                    start_process = time.time()
                    results = pose.process(image)
                    process_time = time.time() - start_process
                    
                    # If processing takes too long, it might be stuck
                    if process_time > 1.0:
                        logger.warning(f"Pose detection took {process_time:.2f}s - may be resource intensive")
                    
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Update cooldowns
                    current_time = time.time()
                    for event_type in event_cooldowns:
                        if event_cooldowns[event_type] > 0:
                            event_cooldowns[event_type] = max(0, event_cooldowns[event_type] - 1)
                    
                    # Try to detect the ball with improved confidence
                    ball_position, ball_confidence, image = detect_ball(image)
                    
                    # Update ball tracker with the detected position
                    ball_detected = False
                    if ball_position and ball_confidence > 0.4:  # Minimum confidence threshold
                        smoothed_ball_pos = ball_tracker.update(ball_position)
                        ball_detected = True
                        
                        # Draw "BALL DETECTED" indicator with confidence
                        conf_color = (0, int(255 * ball_confidence), 0)
                        cv2.putText(image, f"Ball {ball_confidence:.2f}", 
                                    (ball_position[0] - 40, ball_position[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                        
                        # Draw ball trajectory
                        trajectory = ball_tracker.get_trajectory()
                        if len(trajectory) > 2:
                            for i in range(1, len(trajectory)):
                                # Fade color based on how old the point is
                                alpha = (i / len(trajectory))
                                color = (0, int(255 * alpha), int(255 * (1-alpha)))
                                cv2.line(image, trajectory[i-1], trajectory[i], color, 2)
                    
                    # Process pose landmarks if detected
                    if not results.pose_landmarks:
                        cv2.putText(image, "No player detected", (20, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        # Draw pose landmarks
                        try:
                            mp_drawing.draw_landmarks(
                                image, 
                                results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=pose_landmark_drawing_spec,
                                connection_drawing_spec=pose_connection_drawing_spec
                            )
                        except Exception as draw_error:
                            logger.error(f"Error drawing landmarks: {draw_error}")
                        
                        try:
                            # Extract landmarks for football analysis
                            landmarks = results.pose_landmarks.landmark
                            
                            # Check if we have all the landmarks we need
                            required_landmarks = [
                                mp_pose.PoseLandmark.LEFT_ANKLE, 
                                mp_pose.PoseLandmark.RIGHT_ANKLE,
                                mp_pose.PoseLandmark.LEFT_KNEE,
                                mp_pose.PoseLandmark.RIGHT_KNEE,
                                mp_pose.PoseLandmark.LEFT_HIP,
                                mp_pose.PoseLandmark.RIGHT_HIP,
                                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
                            ]
                            
                            missing_landmarks = False
                            for landmark in required_landmarks:
                                if landmark.value >= len(landmarks) or landmarks[landmark.value].visibility < 0.5:
                                    missing_landmarks = True
                                    break
                            
                            if not missing_landmarks:
                                # Get key points for football movements
                                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                
                                # Get additional points for detailed analysis
                                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                                
                                # Calculate angles for leg movements with error handling
                                left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
                                right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
                                
                                # Calculate angle between feet - useful for Feinting detection
                                feet_distance = calculate_distance(left_foot, right_foot)
                                
                                # Convert normalized coordinates to pixel coordinates
                                h, w, _ = image.shape
                                left_ankle_px = (int(left_ankle[0] * w), int(left_ankle[1] * h))
                                right_ankle_px = (int(right_ankle[0] * w), int(right_ankle[1] * h))
                                left_foot_px = (int(left_foot[0] * w), int(left_foot[1] * h))
                                right_foot_px = (int(right_foot[0] * w), int(right_foot[1] * h))
                                
                                # Update foot trackers
                                smoothed_left_foot = left_foot_tracker.update(left_foot_px)
                                smoothed_right_foot = right_foot_tracker.update(right_foot_px)
                                
                                # Show angles for debugging
                                cv2.putText(image, f"L Leg: {left_leg_angle:.1f}", 
                                           (left_ankle_px[0] - 30, left_ankle_px[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.putText(image, f"R Leg: {right_leg_angle:.1f}", 
                                           (right_ankle_px[0] + 10, right_ankle_px[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                # Draw foot speed indicators
                                left_speed = left_foot_tracker.get_speed()
                                right_speed = right_foot_tracker.get_speed()
                                
                                # Visualize foot movement speed
                                left_speed_color = (0, min(255, int(left_speed/5)), 0)
                                right_speed_color = (0, min(255, int(right_speed/5)), 0)
                                
                                cv2.circle(image, left_foot_px, int(5 + min(10, left_speed/50)), left_speed_color, -1)
                                cv2.circle(image, right_foot_px, int(5 + min(10, right_speed/50)), right_speed_color, -1)
                                
                                # ------------------- DRIBBLING SKILL DETECTION -------------------
                                
                                try:
                                    # ----- تعديل خوارزمية اكتشاف المراوغة -----
                                    # 1. Dribble detection - FOCUS ON THIS - تحسين معايير الاكتشاف
                                    if ball_detected and event_cooldowns["dribble"] == 0:
                                        # حساب المسافة بين الكرة وأقرب قدم
                                        left_foot_ball_dist = calculate_distance(left_foot_px, ball_position)
                                        right_foot_ball_dist = calculate_distance(right_foot_px, ball_position)
                                        min_foot_dist = min(left_foot_ball_dist, right_foot_ball_dist)
                                        
                                        # حساب سرعة القدمين
                                        foot_speeds = [left_speed, right_speed]
                                        max_foot_speed = max(foot_speeds)
                                        
                                        # ===== معايير اكتشاف المراوغة الأساسية - أكثر حساسية =====
                                        # المراوغة الأساسية: الكرة قريبة من القدمين + حركة القدمين
                                        basic_dribble = min_foot_dist < 80 and max_foot_speed > 60
                                        
                                        # فحص إضافي لتحديد المراوغات الخاطئة
                                        is_bad_dribble = False
                                        bad_dribble_reason = ""
                                        
                                        # المراوغة تعتبر خاطئة إذا:
                                        if min_foot_dist > 80 and min_foot_dist < 120 and max_foot_speed > 40:
                                            # الكرة بعيدة عن القدمين لكن هناك محاولة للمراوغة
                                            is_bad_dribble = True
                                            bad_dribble_reason = "The ball is too far from feet"
                                        elif min_foot_dist < 80 and max_foot_speed < 40:
                                            # الكرة قريبة لكن حركة القدمين بطيئة جداً
                                            is_bad_dribble = True
                                            bad_dribble_reason = "Foot movement is too slow"
                                        elif left_leg_angle > 160 and right_leg_angle > 160 and min_foot_dist < 100:
                                            # الساقين مستقيمتين أثناء المراوغة (غير مرن)
                                            is_bad_dribble = True
                                            bad_dribble_reason = "Knees are too straight"
                                        
                                        # اكتشاف المراوغة بغض النظر عن انثناء الركبتين
                                        if basic_dribble:
                                            # إذا كان المعيار الأساسي محققاً، نعتبرها مراوغة صحيحة
                                            dribbles_counter += 1
                                            
                                            # حساب مؤشر جودة المراوغة (لكن لا نستخدمه كمعيار للاكتشاف)
                                            quality_score = (1 - (min_foot_dist / 150)) * 0.6  # 60% للمسافة
                                            quality_score += (min(max_foot_speed, 400) / 400) * 0.4  # 40% للسرعة
                                            
                                            # إضافة لقاعدة البيانات
                                            dribble_quality_sum += quality_score
                                            event_cooldowns["dribble"] = 8  # تقليل كولداون لاكتشاف المزيد من المراوغات
                                            
                                            # إضافة بيانات المراوغة
                                            movements_data["dribbles"].append({
                                                "time": current_time - start_time,
                                                "foot_dist": min_foot_dist,
                                                "foot_speed": max_foot_speed,
                                                "quality": round(quality_score * 100)
                                            })
                                            
                                            # عرض الاكتشاف والجودة
                                            quality_percent = int(quality_score * 100)
                                            cv2.putText(image, f"DRIBBLE! {quality_percent}%", (150, 60), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                        elif is_bad_dribble and event_cooldowns["bad_dribble"] == 0:
                                            # تسجيل المراوغة الخاطئة
                                            bad_dribbles_counter += 1
                                            event_cooldowns["dribble"] = 8
                                            event_cooldowns["bad_dribble"] = 15  # Prevent frequent alerts
                                            
                                            # تتبع أسباب المراوغة الخاطئة
                                            if bad_dribble_reason in bad_dribble_reasons:
                                                bad_dribble_reasons[bad_dribble_reason] += 1
                                            else:
                                                bad_dribble_reasons[bad_dribble_reason] = 1
                                                
                                            # عرض تحذير المراوغة الخاطئة
                                            cv2.putText(image, f"BAD DRIBBLE: {bad_dribble_reason}", (120, 60), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                                       
                                            # تعيين التحذير لتشغيل الصوت
                                            warning = bad_dribble_reason
                                            
                                            # Try to make an API call to trigger the alert sound
                                            try:
                                                requests.post('http://localhost:5000/api/play_alert/bad_dribble', 
                                                            timeout=0.1)  # Non-blocking request
                                            except:
                                                # Ignore errors, we don't want to slow down the analysis thread
                                                pass
                                    
                                    # 2. Feinting detection - تحسين اكتشاف الخوزامية - معايير أكثر حساسية
                                    if ball_detected and event_cooldowns["Feinting"] == 0:
                                        # التحقق من حركة القدمين
                                        left_accel = left_foot_tracker.get_acceleration()
                                        right_accel = right_foot_tracker.get_acceleration()
                                        
                                        # الحصول على مواضع القدمين
                                        if len(left_foot_tracker.history) > 3 and len(right_foot_tracker.history) > 3:
                                            # معايير الخوزامية الأساسية: الكرة قريبة + حركة القدمين السريعة
                                            min_foot_dist = min(
                                                calculate_distance(left_foot_px, ball_position),
                                                calculate_distance(right_foot_px, ball_position)
                                            )
                                            
                                            # معايير أكثر حساسية لاكتشاف الخوزامية
                                            basic_Feinting = (
                                                min_foot_dist < 80 and  # الكرة قريبة من القدمين
                                                (left_speed > 80 or right_speed > 80) and  # حركة سريعة للقدمين
                                                (left_accel > 300 or right_accel > 300)  # بعض التسارع للقدمين
                                            )
                                            
                                            if basic_Feinting and current_time - last_Feinting_time > 1.0:
                                                # اكتشفنا خوزامية
                                                Feinting_counter += 1
                                                Feinting_quality_score = 0.7  # جودة افتراضية عالية
                                                Feinting_quality += Feinting_quality_score
                                                last_Feinting_time = current_time
                                                event_cooldowns["Feinting"] = 15
                                                
                                                # تسجيل البيانات
                                                Feinting_sequence.append({
                                                    "time": current_time - start_time,
                                                    "quality": Feinting_quality_score,
                                                    "foot_dist": min_foot_dist,
                                                    "ball_control": 1 - (min_foot_dist / 100)
                                                })
                                                
                                                # إضافة للحركات
                                                movements_data["Feinting"].append({
                                                    "time": current_time - start_time,
                                                    "quality": Feinting_quality_score,
                                                    "ball_control": 1 - (min_foot_dist / 100)
                                                })
                                                
                                                # عرض اكتشاف الخوزامية مع الجودة
                                                quality_percent = int(Feinting_quality_score * 100)
                                                cv2.putText(image, f"Feinting! {quality_percent}%", (220, 80), 
                                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                                            # Add bad feinting detection
                                            elif ball_detected and min_foot_dist > 100 and (left_speed > 80 or right_speed > 80) and event_cooldowns["bad_feint"] == 0:
                                                event_cooldowns["bad_feint"] = 20
                                                bad_dribble_reason = "Your feinting would be more effective with the ball closer to your feet."
                                                
                                                # Display warning on image
                                                cv2.putText(image, "BAD FEINT", (220, 100), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                                                
                                                warning = bad_dribble_reason
                                                
                                                # Try to make an API call to trigger the alert sound
                                                try:
                                                    requests.post('http://localhost:5000/api/play_alert/bad_feint', 
                                                                timeout=0.1)  # Non-blocking request
                                                except:
                                                    # Ignore errors, we don't want to slow down the analysis thread
                                                    pass
                                    
                                    # تحسين التحذيرات لتشغيل التنبيهات الصوتية
                                    if warning == "" and ball_detected and event_cooldowns["warning"] == 0:
                                        # حساب المسافة بين الكرة وأقرب قدم
                                        min_foot_dist = min(
                                            calculate_distance(left_foot_px, ball_position),
                                            calculate_distance(right_foot_px, ball_position)
                                        )
                                        
                                        # Warning conditions
                                        if min_foot_dist > 120:
                                            warning = "Try to keep the ball closer to your feet for better control."
                                            ball_control_score -= 0.1
                                            event_cooldowns["warning"] = 10
                                        elif min_foot_dist > 100 and (left_speed > 50 or right_speed > 50):
                                            warning = "The ball is getting too far from your feet while moving."
                                            ball_control_score -= 0.05
                                            event_cooldowns["warning"] = 10
                                        elif left_leg_angle > 160 and right_leg_angle > 160 and min_foot_dist < 100:
                                            warning = "Keep your knees slightly bent for better balance and control."
                                            performance_score -= 0.1
                                            event_cooldowns["warning"] = 10
                                        elif max(left_speed, right_speed) < 30 and min_foot_dist < 80:
                                            warning = "Try to increase your foot speed during dribbling."
                                            performance_score -= 0.1
                                            event_cooldowns["warning"] = 10
                                    
                                    if warning:
                                        warnings_list.append(warning)
                                    
                                except Exception as event_error:
                                    logger.error(f"Error in event detection: {event_error}")
                                
                                # Keep performance score in valid range
                                performance_score = max(0, min(100, performance_score))
                                ball_control_score = max(0, min(100, ball_control_score))
                            else:
                                # Some landmarks are missing
                                cv2.putText(image, "Some body landmarks not detected", (20, 80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as landmark_error:
                            logger.error(f"Error processing landmarks: {landmark_error}")
                            cv2.putText(image, "Error in landmark processing", (20, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display information on frame
                    elapsed_time = current_time - start_time
                    
                    # Create transparent overlay for stats
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (640, 150), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                    
                    # Top row stats with improved styling - FOCUS ON DRIBBLING ONLY
                    cv2.putText(image, f"Dribbles: {dribbles_counter}", (30, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Add bad dribbles counter
                    cv2.putText(image, f"Bad: {bad_dribbles_counter}", (180, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                    
                    # Add Feinting counter
                    Feinting_color = (0, 255, 255) if Feinting_counter > 0 else (200, 200, 200)
                    cv2.putText(image, f"Feinting: {Feinting_counter}", (280, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, Feinting_color, 2)
                    
                    # Time display
                    cv2.putText(image, f"Time: {elapsed_time:.1f}s", (460, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Warning message if any
                    if warning:
                        cv2.putText(image, f"Tip: {warning}", (30, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Performance scores
                    cv2.putText(image, f"Overall: {int(performance_score)}%", (30, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Ball Control: {int(ball_control_score)}%", (230, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Mode indicator
                    cv2.putText(image, "Mode: Dribbling/Feinting Analysis", (30, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add progress indicator
                    cv2.putText(image, f"Progress: {progress_percent}%", (400, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show tips from GPT - call GPT less frequently to avoid API errors
                    if (dribbles_counter > 0 or Feinting_counter > 0) and (dribbles_counter + Feinting_counter) % 3 == 0 and not gpt_tips_for_overlay:
                        # Try to get tips from GPT
                        try:
                            # Request tips only if we've detected some activity
                            user_msg = (
                                f"Football/Soccer dribbling technique analysis. Details:\n"
                                f"- Dribbles detected: {dribbles_counter}\n"
                                f"- Feinting moves: {Feinting_counter}\n"
                                f"- Current knee bend: Left leg {left_leg_angle:.1f}°, Right leg {right_leg_angle:.1f}°\n"
                                f"- Foot speed: Left {left_speed:.1f}, Right {right_speed:.1f}\n"
                                f"Please provide ONE specific positive tip to enhance the correct dribbling and Feinting technique, "
                                f"focusing only on how to improve the good technique the player already shows. "
                                f"Do not mention weaknesses or mistakes."
                            )
                            
                            # Try to get tips from GPT, with error handling
                            try:
                                response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "user", "content": user_msg}
                                    ],
                                    max_tokens=100  # Limit response size
                                )
                                gpt_tips_for_overlay = response.choices[0].message.content.strip()
                                logger.info(f"Got GPT tip: {gpt_tips_for_overlay[:30]}...")
                            except Exception as gpt_error:
                                logger.error(f"Error calling GPT: {gpt_error}")
                                # نصائح افتراضية أفضل تركز على الإيجابيات
                                dribbling_tips = [
                                    "Continue to keep the ball close to your feet while dribbling to increase control.",
                                    "To improve your dribbling, try increasing the speed at which you switch feet while maintaining the same control.",
                                    "Try changing direction suddenly while dribbling to improve the effectiveness of your dribbling.",
                                    "Maintain a low center of gravity while dribbling for better balance and increased speed.",
                                    "Focus on using the inside of your foot for precise touches while dribbling."
                                ]
                                import random
                                gpt_tips_for_overlay = random.choice(dribbling_tips)
                        except Exception as tip_error:
                            logger.error(f"Error generating tips: {tip_error}")
                    
                    # Show tips on overlay if available
                    if gpt_tips_for_overlay:
                        y0, dy = 420, 20
                        # Add a "Coach Tips" header with background
                        tip_overlay = image.copy()
                        cv2.rectangle(tip_overlay, (10, 380), (630, 460), (0, 0, 0), -1)
                        cv2.addWeighted(tip_overlay, 0.7, image, 0.3, 0, image)
                        
                        cv2.putText(image, "Coach Tips:", (30, 400), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        
                        # Display tips
                        for i, line in enumerate(gpt_tips_for_overlay.split('\n')[:2]):
                            cv2.putText(image, line[:60], (30, y0 + i * dy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Update the output frame with lock
                    with lock:
                        output_frame = image.copy()
                        
                        # Calculate averages
                        avg_dribble_quality = 0
                        if dribbles_counter > 0:
                            avg_dribble_quality = dribble_quality_sum / dribbles_counter
                        
                        # Calculate Feinting quality if applicable
                        Feinting_avg_quality = 0
                        if Feinting_counter > 0:
                            Feinting_avg_quality = Feinting_quality / Feinting_counter
                        
                        # Update analysis results - DRIBBLING FOCUSED with quality metrics
                        current_results = {
                            "dribbles": dribbles_counter,
                            "dribble_quality": round(avg_dribble_quality * 100, 1),
                            "Feinting": Feinting_counter,
                            "Feinting_quality": round(Feinting_avg_quality * 100, 1),
                            "warnings": warning if warning else "",  # Add latest warning for sound alert
                            "tips": gpt_tips_for_overlay,
                            "time": round(elapsed_time, 2),
                            "performance": int(performance_score),
                            "ball_control": int(ball_control_score),
                            "bad_dribbles": bad_dribbles_counter,
                            "bad_dribble_reasons": bad_dribble_reasons,
                            "progress": progress_percent
                        }
                        analysis_results = current_results
                        
                        # Also update the stored analysis
                        if analysis_id in analysis_storage:
                            analysis_storage[analysis_id].update(current_results)
                    
                except Exception as frame_error:
                    logger.error(f"Error processing frame {frame_count}: {frame_error}")
                    traceback.print_exc()
                    
                    # Create error frame
                    error_frame = frame.copy() if frame is not None else np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(error_frame, f"Frame processing error: {str(frame_error)[:50]}", 
                               (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    with lock:
                        output_frame = error_frame
  
                # Slow down processing slightly for smooth display on slower systems
                time.sleep(0.01)
        
        # Release resources
        cap.release()
        logger.info(f"Video analysis completed: {dribbles_counter} dribbles, {Feinting_counter} Feinting moves, {bad_dribbles_counter} bad dribbles")
        
        # Final update with analysis complete
        # Calculate averages
        avg_dribble_quality = 0
        if dribbles_counter > 0:
            avg_dribble_quality = dribble_quality_sum / dribbles_counter
            
        # Calculate Feinting quality
        Feinting_avg_quality = 0
        if Feinting_counter > 0:
           Feinting_avg_quality = Feinting_quality / Feinting_counter
        
        # Create final summary
        with lock:
            analysis_results["analysis_complete"] = True
            analysis_results["bad_dribbles"] = bad_dribbles_counter
            analysis_results["progress"] = 100
            
            # Transform bad_dribble_reasons to a list of tuples for easier handling in frontend
            bad_dribble_reasons_list = []
            for reason, count in bad_dribble_reasons.items():
                bad_dribble_reasons_list.append((reason, count))
            analysis_results["bad_dribble_reasons"] = sorted(bad_dribble_reasons_list, key=lambda x: x[1], reverse=True)
            
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["completed"] = True
                analysis_storage[analysis_id]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                analysis_storage[analysis_id]["bad_dribbles"] = bad_dribbles_counter
                analysis_storage[analysis_id]["bad_dribble_reasons"] = sorted(bad_dribble_reasons_list, key=lambda x: x[1], reverse=True)
                analysis_storage[analysis_id]["progress"] = 100
                
                # Calculate dribbling-specific skill levels
                skill_ratings = {
                    "dribbling": min(10, max(1, int(dribbles_counter * 0.8))),  # Scale more favorably
                    "ball_control": min(10, max(1, round(ball_control_score/10))),
                    "Feinting_technique": min(10, max(1, round(Feinting_counter * 0.7))),  # Scale more favorably
                    "agility": min(10, max(1, round((performance_score / 10))))
                }
                
                analysis_storage[analysis_id]["skills"] = skill_ratings
                analysis_storage[analysis_id]["movement_data"] = movements_data
                
                # Identify top errors with counts but only for major issues
                if warnings_list:
                    warnings_count = {}
                    for warning in warnings_list:
                        if warning in warnings_count:
                            warnings_count[warning] += 1
                        else:
                            warnings_count[warning] = 1
                    
                    # Get top 2 most common errors
                    top_errors = sorted(warnings_count.items(), key=lambda x: x[1], reverse=True)[:2]
                    analysis_storage[analysis_id]["top_errors"] = [{"error": e, "count": c} for e, c in top_errors]
                else:
                    analysis_storage[analysis_id]["top_errors"] = []
                    
                # Add exercise type
                analysis_storage[analysis_id]["exercise_type"] = "dribbling"
                
                # Generate final summary
                try:
                    # Try to get a final summary from GPT
                    if openai.api_key:
                        summary_msg = (
                            f"Please provide a concise positive summary for a football player who completed a dribbling training session, focusing on Feinting technique.\n\n"
                            f"Session statistics:\n"
                            f"- Duration: {round(elapsed_time, 1)} seconds\n"
                            f"- Dribbles: {dribbles_counter}\n"
                            f"- Dribble quality score: {round(avg_dribble_quality * 100)}%\n"
                            f"- Bad dribbles: {bad_dribbles_counter}\n"
                            f"- Feinting moves: {Feinting_counter}\n"
                            f"- Feinting quality: {round(Feinting_avg_quality * 100)}%\n"
                            f"- Ball control: {int(ball_control_score)}%\n\n"
                            f"Create a positive and encouraging summary focusing on strengths and correct techniques. "
                            f"Then provide 2-3 specific tips for further enhancing these already good techniques. "
                            f"Your response should be upbeat and focus only on what the player did well, assuming they have good dribbling skills."
                        )
                        
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "user", "content": summary_msg}
                                ],
                                max_tokens=200
                            )
                            final_summary = response.choices[0].message.content.strip()
                            analysis_storage[analysis_id]["summary"] = final_summary
                            
                            # Generate audio for the summary
                            try:
                                # Generate spoken report audio
                                report_audio_file = f"audio/reports/report_{analysis_id}.mp3"
                                text_to_speech(final_summary, report_audio_file)
                                analysis_storage[analysis_id]["summary_audio"] = report_audio_file
                            except Exception as audio_error:
                                logger.error(f"Error generating summary audio: {audio_error}")
                        except Exception as gpt_error:
                            logger.error(f"Error generating GPT summary: {gpt_error}")
                            # Create a positive fallback summary for dribbling
                            fallback_summary = (
                                f"You showed outstanding performance with {dribbles_counter} Strong dribble and {Feinting_counter} Good technique Feinting movement "
                                f"Your ball control was {int(ball_control_score)}%. "
                                f"\n\nTo develop, focus on: "
                                f"\n1. Keeping your knees slightly bent for more flexibility and control while dribbling."
                                f"\n2. Accelerating your footwork during the dribbling while maintaining the same level of accuracy."
                                f"\n3. Developing your ability to change direction suddenly to increase dribbling effectiveness."
                            )
                            analysis_storage[analysis_id]["summary"] = fallback_summary
                            
                            # Generate audio for fallback summary
                            try:
                                report_audio_file = f"audio/reports/report_{analysis_id}.mp3"
                                text_to_speech(fallback_summary, report_audio_file)
                                analysis_storage[analysis_id]["summary_audio"] = report_audio_file
                            except Exception as audio_error:
                                logger.error(f"Error generating fallback summary audio: {audio_error}")
                    else:
                        # No API key available
                        simple_summary = (
                            f"Dribbling analysis complete. {dribbles_counter} showed good dribbling and "
                            f"{Feinting_counter} Feinting movement over {round(elapsed_time)} seconds with a control level of {int(ball_control_score)}%. "
                            f"Continue to keep the ball close to your feet and develop speed in changing feet in Feinting!"
                        )
                        analysis_storage[analysis_id]["summary"] = simple_summary
                        
                        # Generate audio for simple summary
                        try:
                            report_audio_file = f"audio/reports/report_{analysis_id}.mp3"
                            text_to_speech(simple_summary, report_audio_file)
                            analysis_storage[analysis_id]["summary_audio"] = report_audio_file
                        except Exception as audio_error:
                            logger.error(f"Error generating simple summary audio: {audio_error}")
                except Exception as summary_error:
                    logger.error(f"Error creating summary: {summary_error}")
                    default_summary = "Dribbling analysis completed successfully. You showed excellent ball control skills!"
                    analysis_storage[analysis_id]["summary"] = default_summary
                    
                    # Generate audio for default summary
                    try:
                        report_audio_file = f"audio/reports/report_{analysis_id}.mp3"
                        text_to_speech(default_summary, report_audio_file)
                        analysis_storage[analysis_id]["summary_audio"] = report_audio_file
                    except Exception as audio_error:
                        logger.error(f"Error generating default summary audio: {audio_error}")
        
    except Exception as e:
        logger.error(f"Error in video analysis: {e}")
        traceback.print_exc()
        with lock:
            analysis_results = {"error": str(e), "progress": 0}
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["error"] = str(e)
    finally:
        is_analyzing = False
        if 'cap' in locals() and cap is not None:
            cap.release()
        logger.info("Analysis thread completed")

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Route to handle video upload and start analysis."""
    global is_analyzing, analysis_storage
    
    try:
        # Check if already analyzing
        if is_analyzing:
            return jsonify({"error": "Already analyzing a video. Please wait."}), 400
        
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({"error": "No video file provided."}), 400
        
        # Validate video file
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
            return jsonify({"error": "Invalid video format. Please upload MP4, AVI, MOV, WMV, or MKV files."}), 400
        
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save the uploaded video
        video_path = f'uploaded_video_{analysis_id}.mp4'
        video_file.save(video_path)
        
        # Verify the video can be opened
        try:
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                os.remove(video_path)
                return jsonify({"error": "The uploaded file is not a valid video."}), 400
            
            # Get basic video info
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                test_cap.release()
                os.remove(video_path)
                return jsonify({"error": "Invalid video format or corrupted file."}), 400
                
            # Check video is not too long
            duration = frame_count / fps if fps > 0 else 0
            if duration > 300:  # 5 minutes max
                test_cap.release()
                os.remove(video_path)
                return jsonify({"error": "Video too long. Please limit to 5 minutes maximum."}), 400
                
            test_cap.release()
        except Exception as e:
            logger.error(f"Error validating video: {e}")
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({"error": f"Could not process video: {str(e)}"}), 400
        
        # Initialize analysis storage
        analysis_storage[analysis_id] = {
            "id": analysis_id,
            "video_path": video_path,
            "exercise_type": "dribbling",  # Always dribbling
            "completed": False,
            "progress": 0
        }
        
        # Start analysis in a separate thread
        thread = threading.Thread(target=analyze_video_thread, args=(video_path, analysis_id))
        thread.daemon = True
        thread.start()
        
        # Store the analysis ID in session
        session['current_analysis'] = analysis_id
        
        return jsonify({
            "message": "Dribbling video analysis started", 
            "stream_url": "/video_feed", 
            "analysis_id": analysis_id
        })
    except Exception as e:
        logger.error(f"Error in analyze_video route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/finish_analysis')
def finish_analysis():
    """Redirect to coach view with current analysis"""
    try:
        analysis_id = session.get('current_analysis')
        if not analysis_id:
            return redirect('/coach_dribbling.html')
        
        return redirect(url_for('serve_html', path=f'coach_dribbling.html?analysis_id={analysis_id}'))
    except Exception as e:
        logger.error(f"Error in finish_analysis: {e}")
        return redirect(url_for('serve_html', path='coach_dribbling.html'))

@app.route('/clear_old_analysis', methods=['POST'])
def clear_old_analysis():
    """Clean up old analysis data to free up memory"""
    try:
        # Keep only the 10 most recent analyses
        if len(analysis_storage) > 10:
            # Sort by timestamp (or creation time if no timestamp)
            sorted_analyses = sorted(
                analysis_storage.items(),
                key=lambda x: x[1].get('timestamp', '1970-01-01')
            )
            
            # Remove oldest analyses and their video files
            for analysis_id, analysis in sorted_analyses[:-10]:
                if 'video_path' in analysis and os.path.exists(analysis['video_path']):
                    try:
                        os.remove(analysis['video_path'])
                    except:
                        pass
                        
                if 'thumbnail' in analysis and os.path.exists(analysis['thumbnail']):
                    try:
                        os.remove(analysis['thumbnail'])
                    except:
                        pass
                        
                # Remove any associated audio files
                audio_file = f"audio/reports/report_{analysis_id}.mp3"
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                        
                del analysis_storage[analysis_id]
        
        return jsonify({"success": True, "remaining": len(analysis_storage)})
    except Exception as e:
        logger.error(f"Error clearing old analyses: {e}")
        return jsonify({"error": str(e)}), 500

# Make sure we have a static directory for audio files
if not os.path.exists('static'):
    os.makedirs('static')

# Make sure audio directories exist
os.makedirs('audio/reports', exist_ok=True)
os.makedirs('audio/alerts', exist_ok=True)

# Create a default alert sound if one doesn't exist
if not os.path.exists('static/alert.mp3'):
    try:
        # Try to download a default alert sound
        import urllib.request
        urllib.request.urlretrieve(
            "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
            "static/alert.mp3"
        )
        logger.info("Downloaded default alert sound")
    except Exception as e:
        logger.error(f"Could not download alert sound: {e}")

if __name__ == '__main__':
    # Make sure the necessary files exist
    for required_file in ['home_dribbling.html', 'coach_dribbling.html']:
        if not os.path.exists(required_file):
            logger.error(f"Required file {required_file} not found")
            print(f"ERROR: Required file {required_file} not found in {os.getcwd()}")
            print("Make sure these HTML files are in the same directory as app.py")
    
    # Generate alert sounds at startup
    try:
        bad_dribble_text = "Keep the ball closer to your feet! Try to maintain the ball within 60 centimeters."
        bad_dribble_file = "audio/alerts/bad_dribble.mp3"
        text_to_speech(bad_dribble_text, bad_dribble_file)
        
        bad_feint_text = "Improve your feinting by keeping the ball closer to your feet. Try quicker foot movements."
        bad_feint_file = "audio/alerts/bad_feint.mp3"
        text_to_speech(bad_feint_text, bad_feint_file)
        
        logger.info("Pre-generated alert sounds at startup")
    except Exception as e:
        logger.error(f"Error pre-generating alert sounds: {e}")
    
    logger.info("Starting football dribbling analysis server")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)