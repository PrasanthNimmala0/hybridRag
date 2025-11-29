import os
import subprocess as sp
import requests
import wikipedia
import pywhatkit as kit
from email.message import EmailMessage
import smtplib
from decouple import config
from datetime import datetime, date
import gradio as gr
import tempfile
import webbrowser
from urllib.parse import quote
import asyncio
import logging
import pandas as pd
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from urllib import request as url_request
import json
import shutil
import atexit

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool

# Groq for STT
from groq import Groq

# Tavily for real-time search
from tavily import TavilyClient

# TTS
from gtts import gTTS

# Emotion Detection imports
import cv2
import numpy as np
from PIL import Image
import threading
import queue

# TensorFlow for emotion detection
import tensorflow as tf

# TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob not available. Install with: pip install textblob")

# ==============================================================================
# ------------------ SECTION 1: CONFIGURATION & INITIALIZATION -----------------
# ==============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check GPU availability
gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    print("✅ GPU available for TensorFlow")
else:
    print("⚠️ GPU not available, using CPU")

# Load API Keys and Config from .env file
GROQ_API_KEY = config("GROQ_API_KEY")
TAVILY_API_KEY = config("TAVILY_API_KEY")
USERNAME = config('USER', default='User')
BOTNAME = config('BOTNAME', default='Friday')
NEWS_API_KEY = config("NEWS_API_KEY")
OPENWEATHER_APP_ID = config("OPENWEATHER_APP_ID")
TMDB_API_KEY = config("TMDB_API_KEY")
EMAIL = config("EMAIL")
PASSWORD = config("PASSWORD")

# Initialize Groq Client for STT
groq_client = Groq(api_key=GROQ_API_KEY)

# Global Conversation History for RAG Bot
conversation_history = []
MAX_HISTORY = 10

# WhatsApp Conversation State Management
whatsapp_state = {
    'active': False,
    'number': None,
    'stage': None
}

# Email Conversation State Management
email_state = {
    'active': False,
    'email': None,
    'subject': None,
    'stage': None
}

# Emotion State Management
emotion_state = {
    'current_emotion': 'Neutral',
    'confidence': 0.0,
    'history': [],
    'is_camera_running': False
}

# Static Emotion State for uploaded images
static_emotion_state = {
    'history': []
}

# NEW: Emotion-to-Chat Trigger State
emotion_chat_trigger = {
    'triggered': False,
    'emotion': None,
    'last_consumed_timestamp': 0 # Timestamp of when the last trigger was consumed
}
EMOTION_TRIGGER_THRESHOLD = 0.50  # 50% confidence
EMOTION_TRIGGER_COOLDOWN = 120 # 120 seconds cooldown between proactive questions


# Initialize LLM and Embeddings for LlamaIndex
def initialize_llama_index():
    """Initialize LlamaIndex with Groq LLM and HuggingFace embeddings"""
    llm = GroqLLM(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    return llm, embed_model

# ==============================================================================
# ---------------- SECTION 2: REAL-TIME EMOTION DETECTION ----------------------
# ==============================================================================

class EmotionDetector:
    """Real-time Emotion Detection using OpenCV and TensorFlow"""
    
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = None
        self.face_cascade = None
        self.is_running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.emotion_queue = queue.Queue(maxsize=50)
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0
        self.face_detected = False
        self._frame_counter = 0

        # Adaptive thresholds and smoothing buffers
        self._recent_face_sizes: deque = deque(maxlen=30)
        self._prediction_window: deque = deque(maxlen=5)

        # Initialize components
        self._load_model(model_path)
        self._load_face_cascade()
    
    def _load_model(self, model_path: str = None):
        """Load the emotion detection model"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.logger.info(f"✅ Loaded emotion model from {model_path}")
            else:
                # Try to load a pretrained FER2013 model automatically
                self.model = self._load_pretrained_default_model()
                if self.model is not None:
                    self.logger.info("✅ Loaded pretrained FER2013 emotion model")
                else:
                    # Create a simple CNN model for emotion detection
                    self.model = self._create_default_model()
                    self.logger.warning("⚠️ Using UNTRAINED emotion model! Predictions will be random.")
                    self.logger.warning("⚠️ The app will auto-download a pretrained model on first run.")
        except Exception as e:
            self.logger.error(f"❌ Error loading emotion model: {e}")
            self.model = self._create_default_model()
            self.logger.warning("⚠️ Using UNTRAINED fallback model!")
    
    def _create_default_model(self):
        """Create a default CNN model for emotion detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        model.build(input_shape=(None, 48, 48, 1))
        return model

    def _load_pretrained_default_model(self):
        """Attempt to load a pretrained FER2013 model with progress bar"""
        try:
            models_dir = Path('models')
            models_dir.mkdir(parents=True, exist_ok=True)

            filename = 'fer2013_mini_XCEPTION.hdf5'
            local_path = models_dir / filename

            if not local_path.exists():
                url = (
                    'https://github.com/oarriaga/face_classification/raw/master/'
                    'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
                )
                tmp_path = str(local_path) + '.download'
                self.logger.info(f"📥 Downloading pretrained emotion model...")
                
                try:
                    import urllib.request
                    
                    def reporthook(block_num, block_size, total_size):
                        """Progress callback"""
                        downloaded = block_num * block_size
                        percent = min(downloaded * 100 / total_size, 100)
                        bar_length = 40
                        filled = int(bar_length * percent / 100)
                        bar = '█' * filled + '-' * (bar_length - filled)
                        print(f"\r📥 Downloading: |{bar}| {percent:.1f}%", end='', flush=True)
                    
                    urllib.request.urlretrieve(url, tmp_path, reporthook)
                    print()  # New line after progress
                    os.replace(tmp_path, local_path)
                    self.logger.info(f"✅ Downloaded to {local_path}")
                except Exception as de:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
                    self.logger.warning(f"❌ Download failed: {de}")
                    return None

            model = tf.keras.models.load_model(str(local_path))
            try:
                input_shape = model.input_shape
                if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
                    height = input_shape[1]
                    self._pretrained_input_size = height if isinstance(height, int) and height > 0 else 64
                else:
                    self._pretrained_input_size = 64
            except:
                self._pretrained_input_size = 64
            return model
        except Exception as e:
            self.logger.warning(f"⚠️ Pretrained model load failed: {e}")
            return None
    
    def _load_face_cascade(self):
        """Load OpenCV face cascade classifier"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade")
            self.logger.info("✅ Face cascade loaded successfully")
        except Exception as e:
            self.logger.error(f"❌ Error loading face cascade: {e}")
            self.face_cascade = None
    
    def preprocess_face(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face region for emotion detection"""
        try:
            if len(face_roi.shape) == 3:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_roi

            equalized = cv2.equalizeHist(face_gray)
            input_size = getattr(self, "_pretrained_input_size", 48)
            face_resized = cv2.resize(equalized, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            face_normalized = face_resized.astype("float32") / 255.0
            face_reshaped = face_normalized.reshape(1, input_size, input_size, 1)

            return face_reshaped
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {e}")
            return None
    
    def _smooth_prediction(self, emotion: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to reduce prediction jitter"""
        self._prediction_window.append((emotion, confidence))

        if len(self._prediction_window) < 2:
            return emotion, confidence

        weights = np.linspace(1.0, 0.4, num=len(self._prediction_window))
        weight_sum = weights.sum()
        aggregated: Dict[str, float] = {}

        for weight, (label, conf) in zip(reversed(weights), self._prediction_window):
            aggregated[label] = aggregated.get(label, 0.0) + weight * conf

        smoothed_emotion = max(aggregated, key=aggregated.get)
        smoothed_confidence = aggregated[smoothed_emotion] / weight_sum

        return smoothed_emotion, float(smoothed_confidence)

    def detect_emotion(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Detect emotion from face region"""
        try:
            if self.model is None:
                return "Neutral", 0.5

            processed_face = self.preprocess_face(face_roi)
            if processed_face is None:
                return "Neutral", 0.0

            predictions = self.model.predict(processed_face, verbose=0)
            emotion_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][emotion_index])

            emotion = self.emotion_labels[emotion_index]
            return self._smooth_prediction(emotion, confidence)

        except Exception as e:
            self.logger.error(f"Error detecting emotion: {e}")
            return "Neutral", 0.0
    
    def _get_dynamic_detection_params(self) -> Dict:
        """Adjust face detection sensitivity based on recent face sizes"""
        if not self._recent_face_sizes:
            return {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)}

        average_size = sum(self._recent_face_sizes) / len(self._recent_face_sizes)
        min_size = max(24, int(average_size * 0.6))

        if average_size < 50:
            scale_factor = 1.05
            min_neighbors = 3
        elif average_size < 120:
            scale_factor = 1.08
            min_neighbors = 4
        else:
            scale_factor = 1.1
            min_neighbors = 5

        return {"scaleFactor": scale_factor, "minNeighbors": min_neighbors, "minSize": (min_size, min_size)}

    def process_frame(self, frame: np.ndarray, transcript: Optional[str] = None) -> Dict:
        """Process a single frame for emotion detection"""
        try:
            result = {
                "emotion": "Neutral",
                "confidence": 0.0,
                "face_detected": False,
                "face_bbox": None,
                "frame": frame.copy(),
            }

            if self.face_cascade is None:
                return result

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            params = self._get_dynamic_detection_params()
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=params["scaleFactor"],
                minNeighbors=params["minNeighbors"],
                minSize=params["minSize"],
            )

            if len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                self._recent_face_sizes.append(max(w, h))
                face_roi = gray[y : y + h, x : x + w]
                emotion, confidence = self.detect_emotion(face_roi)

                result.update({
                    "emotion": emotion,
                    "confidence": confidence,
                    "face_detected": True,
                    "face_bbox": largest_face,
                })

                # Draw bounding box and emotion on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion}: {confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                result["frame"] = frame

            return result

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return {
                "emotion": "Neutral",
                "confidence": 0.0,
                "face_detected": False,
                "face_bbox": None,
                "frame": frame,
            }
    
    def _try_open_camera(self, index: int) -> bool:
        """Try to open camera with best backend"""
        try:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(index)
            return self.cap.isOpened()
        except Exception as e:
            self.logger.error(f"Error opening camera index {index}: {e}")
            return False

    def find_working_camera(self, max_index: int = 5) -> int:
        """Scan camera indices and return a working one"""
        for idx in range(0, max_index + 1):
            if self._try_open_camera(idx):
                self.logger.info(f"Camera index {idx} opened successfully")
                return idx
            if hasattr(self, 'cap'):
                try:
                    self.cap.release()
                except:
                    pass
        self.logger.error("No working camera found")
        return -1

    def start_camera(self, camera_index: int = None) -> bool:
        """Start camera capture. If camera_index is None, auto-detect."""
        try:
            selected_index = camera_index
            if selected_index is None:
                selected_index = self.find_working_camera()
                if selected_index < 0:
                    return False
            else:
                if not self._try_open_camera(selected_index):
                    self.logger.warning(f"Camera index {selected_index} failed, attempting auto-detect")
                    selected_index = self.find_working_camera()
                    if selected_index < 0:
                        return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # Try to set MJPG for better throughput
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            
            self.is_running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("✅ Camera started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting camera: {e}")
            return False
    
    def _camera_loop(self):
        """Main camera processing loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    continue
                
                # Throttle heavy processing to reduce CPU load
                self._frame_counter = (self._frame_counter + 1) % 3
                if self._frame_counter == 0:
                    # Full processing with detection/prediction
                    result = self.process_frame(frame)
                else:
                    # Lightweight update: pass-through frame and keep last state
                    result = {
                        'emotion': self.current_emotion,
                        'confidence': self.current_confidence,
                        'face_detected': self.face_detected,
                        'face_bbox': None,
                        'frame': frame
                    }
                
                # Update current state
                self.current_emotion = result['emotion']
                self.current_confidence = result['confidence']
                self.face_detected = result['face_detected']
                
                # Add to queues
                if not self.frame_queue.full():
                    self.frame_queue.put(result['frame'])
                
                if not self.emotion_queue.full():
                    self.emotion_queue.put({
                        'emotion': result['emotion'],
                        'confidence': result['confidence'],
                        'face_detected': result['face_detected'],
                        'timestamp': time.time()
                    })
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.logger.error(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest processed frame"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
        except queue.Empty:
            return None
    
    def get_emotion_history(self, max_items: int = 10) -> List[Dict]:
        """Get recent emotion detection history"""
        try:
            emotions = []
            temp_queue = queue.Queue()
            
            # Extract items from emotion queue
            while not self.emotion_queue.empty():
                item = self.emotion_queue.get_nowait()
                emotions.append(item)
                temp_queue.put(item)
            
            # Put items back
            while not temp_queue.empty():
                self.emotion_queue.put(temp_queue.get_nowait())
            
            return emotions[-max_items:] if emotions else []
            
        except Exception as e:
            self.logger.error(f"Error getting emotion history: {e}")
            return []
    
    def get_current_state(self) -> Dict:
        """Get current emotion detection state"""
        return {
            'emotion': self.current_emotion,
            'confidence': self.current_confidence,
            'face_detected': self.face_detected,
            'is_running': self.is_running
        }
    
    def stop_camera(self):
        """Stop camera capture"""
        try:
            self.is_running = False
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            
            if hasattr(self, 'processing_thread'):
                self.processing_thread.join(timeout=2)
            
            self.logger.info("✅ Camera stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping camera: {e}")
    
    def get_emotion_statistics(self) -> Dict:
        """Get emotion detection statistics"""
        try:
            emotions = self.get_emotion_history(max_items=100)
            if not emotions:
                return {}
            
            emotion_counts = {}
            total_confidence = 0
            
            for item in emotions:
                emotion = item['emotion']
                confidence = item['confidence']
                
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = {'count': 0, 'total_confidence': 0}
                
                emotion_counts[emotion]['count'] += 1
                emotion_counts[emotion]['total_confidence'] += confidence
                total_confidence += confidence
            
            # Calculate averages
            for emotion in emotion_counts:
                count = emotion_counts[emotion]['count']
                emotion_counts[emotion]['avg_confidence'] = emotion_counts[emotion]['total_confidence'] / count
                emotion_counts[emotion]['percentage'] = (count / len(emotions)) * 100
            
            return emotion_counts
            
        except Exception as e:
            self.logger.error(f"Error calculating emotion statistics: {e}")
            return {}
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_camera()

# Initialize global emotion detector
emotion_detector = EmotionDetector()

# ==============================================================================
# ---------------- SECTION 3: STATIC EMOTION DETECTION (UPLOAD) ----------------
# ==============================================================================

class StaticEmotionDetector:
    """Static emotion detection for uploaded images"""
    
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_colors = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Surprise': (0, 255, 255),
            'Fear': (128, 0, 128),
            'Disgust': (0, 128, 128),
            'Neutral': (255, 255, 255)
        }
        
        # Use the same model as real-time detector
        self.model = emotion_detector.model
        self.face_cascade = emotion_detector.face_cascade
    
    def detect_emotion_from_image(self, image):
        """Detect emotion from uploaded image"""
        global emotion_chat_trigger
        
        if image is None:
            return None, "⚠️ No image provided. Please upload an image."
        
        temp_file = None
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Detect faces
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None, "⚠️ No face detected in the image. Please upload an image with a clear face."
            
            # Process largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect emotion using the same preprocessing as real-time
            processed_face = emotion_detector.preprocess_face(face_roi)
            if processed_face is None:
                return None, "⚠️ Error processing face"
            
            predictions = self.model.predict(processed_face, verbose=0)
            emotion_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][emotion_index])
            emotion = self.emotion_labels[emotion_index]
            
            # Get all emotion scores
            all_emotions = {label: float(predictions[0][i]) * 100 for i, label in enumerate(self.emotion_labels)}
            
            # Annotate image
            annotated_image = image_bgr.copy()
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            
            label = f"{emotion}: {confidence:.2%}"
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_image,
                (x, label_y - text_height - 5),
                (x + text_width, label_y + 5),
                color,
                -1
            )
            
            cv2.putText(
                annotated_image, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            
            # Convert back to RGB for Gradio
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            
            # Enhanced response with emoji
            emoji_map = {
                'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
                'Surprise': '😮', 'Fear': '😨', 'Disgust': '🤢', 'Neutral': '😐'
            }
            
            response_text = f"""
# {emoji_map.get(emotion, '😐')} **Emotion Detection Results**

👤 **Faces Detected:** {len(faces)}
{emoji_map.get(emotion, '😐')} **Dominant Emotion:** **{emotion.upper()}**
📊 **Confidence:** **{confidence:.1%}**

---

## 📈 All Emotion Scores:

"""
            
            for em, score in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True):
                bar_length = int(score / 5)
                bar = '█' * bar_length + '░' * (20 - bar_length)
                emoji = emoji_map.get(em, '😐')
                response_text += f"{emoji} **{em:<10}** `{bar}` {score:.1f}%\n"
            
            # NEW: Emotion Trigger Logic
            if (not emotion_chat_trigger['triggered'] and
                confidence > EMOTION_TRIGGER_THRESHOLD and 
                emotion != 'Neutral'):

                current_time = time.time()
                if (current_time - emotion_chat_trigger['last_consumed_timestamp']) > EMOTION_TRIGGER_COOLDOWN:
                    emotion_chat_trigger['triggered'] = True
                    emotion_chat_trigger['emotion'] = emotion
                    print(f"🔥 Static emotion trigger set: {emotion}")

            # Display the message if a trigger is pending
            if emotion_chat_trigger['triggered']:
                response_text += "\n\n---\n\n**✨ I have a question for you in the Chat tab!**"

            # Update static emotion state
            global static_emotion_state
            static_emotion_state['history'].append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Keep last 50 records
            if len(static_emotion_state['history']) > 50:
                static_emotion_state['history'] = static_emotion_state['history'][-50:]
            
            return annotated_pil, response_text
        
        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ Error detecting emotion: {str(e)}"
        
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

# Initialize static emotion detector
static_emotion_detector = StaticEmotionDetector()

# ==============================================================================
# ---------------- SECTION 4: CORE UTILITY FUNCTIONS ---------------------------
# ==============================================================================

# OS Operations
paths = {
    'notepad': "C:\\Program Files\\Notepad++\\notepad++.exe",
    'discord': "C:\\Users\\Public\\AppData\\Local\\Discord\\app-1.0.9003\\Discord.exe",
    'calculator': "C:\\Windows\\System32\\calc.exe"
}

def open_notepad(): 
    try:
        os.startfile(paths['notepad'])
    except Exception as e:
        print(f"Notepad Error: {e}")

def open_discord(): 
    try:
        os.startfile(paths['discord'])
    except Exception as e:
        print(f"Discord Error: {e}")

def open_cmd(): 
    os.system('start cmd')

def open_camera(): 
    sp.run('start microsoft.windows.camera:', shell=True)

def open_calculator(): 
    try:
        sp.Popen(paths['calculator'])
    except Exception as e:
        print(f"Calculator Error: {e}")

# Online Operations
def find_my_ip(): 
    try:
        return requests.get('https://api64.ipify.org?format=json', timeout=5).json()["ip"]
    except Exception as e:
        print(f"IP Error: {e}")
        return "Unable to fetch IP"

def search_on_wikipedia(query): 
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Could not find information about {query}"

def play_on_youtube(video): 
    try:
        kit.playonyt(video)
    except Exception as e:
        print(f"YouTube Error: {e}")

def search_on_google(query): 
    try:
        kit.search(query)
    except Exception as e:
        print(f"Google Search Error: {e}")

def search_on_website(query, website):
    """Search on specific websites like Ajio, Amazon, Flipkart, etc."""
    try:
        encoded_query = quote(query)
        
        search_urls = {
            'ajio': f'https://www.ajio.com/search/?text={encoded_query}',
            'amazon': f'https://www.amazon.in/s?k={encoded_query}',
            'flipkart': f'https://www.flipkart.com/search?q={encoded_query}',
            'myntra': f'https://www.myntra.com/{encoded_query}',
            'meesho': f'https://www.meesho.com/search?q={encoded_query}',
            'snapdeal': f'https://www.snapdeal.com/search?keyword={encoded_query}',
            'nykaa': f'https://www.nykaa.com/search/result/?q={encoded_query}',
            'ebay': f'https://www.ebay.in/sch/i.html?_nkw={encoded_query}',
            'shopclues': f'https://www.shopclues.com/search?q={encoded_query}',
            'paytmmall': f'https://paytmmall.com/shop/search?q={encoded_query}',
            'tatacliq': f'https://www.tatacliq.com/search/?searchText={encoded_query}',
            'jiomart': f'https://www.jiomart.com/search/{encoded_query}',
            'bigbasket': f'https://www.bigbasket.com/ps/?q={encoded_query}',
            'swiggy': f'https://www.swiggy.com/search?query={encoded_query}',
            'zomato': f'https://www.zomato.com/search?q={encoded_query}',
            'bookmyshow': f'https://in.bookmyshow.com/explore/home/{encoded_query}',
            'linkedin': f'https://www.linkedin.com/search/results/all/?keywords={encoded_query}',
            'twitter': f'https://twitter.com/search?q={encoded_query}',
            'instagram': f'https://www.instagram.com/explore/tags/{encoded_query.replace(" ", "")}/',
            'reddit': f'https://www.reddit.com/search/?q={encoded_query}',
            'pinterest': f'https://www.pinterest.com/search/pins/?q={encoded_query}',
            'stackoverflow': f'https://stackoverflow.com/search?q={encoded_query}',
            'github': f'https://github.com/search?q={encoded_query}',
            'netflix': f'https://www.netflix.com/search?q={encoded_query}',
            'hotstar': f'https://www.hotstar.com/in/search?q={encoded_query}',
            'primevideo': f'https://www.primevideo.com/search?phrase={encoded_query}',
        }
        
        website_lower = website.lower()
        url = search_urls.get(website_lower)
        
        if url:
            webbrowser.open(url)
            return True
        else:
            fallback_url = f'https://www.{website_lower}.com/search?q={encoded_query}'
            webbrowser.open(fallback_url)
            return True
            
    except Exception as e:
        print(f"Website Search Error: {e}")
        return False

def open_website(website):
    """Open a specific website"""
    try:
        website_urls = {
            'ajio': 'https://www.ajio.com',
            'amazon': 'https://www.amazon.in',
            'flipkart': 'https://www.flipkart.com',
            'myntra': 'https://www.myntra.com',
            'meesho': 'https://www.meesho.com',
            'snapdeal': 'https://www.snapdeal.com',
            'nykaa': 'https://www.nykaa.com',
            'gmail': 'https://mail.google.com',
            'youtube': 'https://www.youtube.com',
            'facebook': 'https://www.facebook.com',
            'instagram': 'https://www.instagram.com',
            'twitter': 'https://www.twitter.com',
            'linkedin': 'https://www.linkedin.com',
            'netflix': 'https://www.netflix.com',
            'hotstar': 'https://www.hotstar.com',
            'github': 'https://www.github.com',
            'reddit': 'https://www.reddit.com',
            'pinterest': 'https://www.pinterest.com',
        }
        
        website_lower = website.lower()
        url = website_urls.get(website_lower, f'https://www.{website_lower}.com')
        webbrowser.open(url)
        return True
    except Exception as e:
        print(f"Website Open Error: {e}")
        return False

def send_whatsapp_message(number, message): 
    """Send WhatsApp message with validation"""
    try:
        # Validate inputs
        if not number.isdigit() or len(number) != 10:
            raise ValueError("Invalid phone number format. Must be 10 digits.")
        
        if not message.strip():
            raise ValueError("Message cannot be empty")
        
        kit.sendwhatmsg_instantly(f"+91{number}", message)
        return True, "✅ Message sent successfully"
    except Exception as e:
        error_msg = str(e)
        if "Please open WhatsApp" in error_msg or "WhatsApp Web" in error_msg:
            return False, "❌ WhatsApp Web not logged in. Please scan QR code first."
        elif "wait" in error_msg.lower():
            return False, "❌ Rate limited. Wait 30 seconds and try again."
        else:
            return False, f"❌ Error: {error_msg}"

def send_email(receiver_address, subject, message):
    """Send email with validation"""
    try:
        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, receiver_address):
            raise ValueError("Invalid email format")
        
        if not subject.strip() or not message.strip():
            raise ValueError("Subject and message cannot be empty")
        
        email = EmailMessage()
        email['To'] = receiver_address
        email["Subject"] = subject
        email['From'] = EMAIL
        email.set_content(message)
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.starttls()
        s.login(EMAIL, PASSWORD)
        s.send_message(email)
        s.close()
        return True, "✅ Email sent successfully"
    except Exception as e:
        return False, f"❌ Email Error: {str(e)}"

def get_latest_news():
    try:
        res = requests.get(
            f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}&category=general", 
            timeout=5
        ).json()
        return [a["title"] for a in res.get("articles", [])][:5]
    except Exception as e:
        print(f"News Error: {e}")
        return ["Unable to fetch news"]

def get_weather_report(city):
    try:
        res = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_APP_ID}&units=metric", 
            timeout=5
        ).json()
        weather = res["weather"][0]["main"]
        temperature = res["main"]["temp"]
        feels_like = res["main"]["feels_like"]
        return weather, f"{temperature}℃", f"{feels_like}℃"
    except Exception as e:
        print(f"Weather Error: {e}")
        return "Unknown", "N/A", "N/A"

def get_trending_movies():
    try:
        res = requests.get(
            f"https://api.themoviedb.org/3/trending/movie/day?api_key={TMDB_API_KEY}", 
            timeout=5
        ).json()
        return [r["original_title"] for r in res.get("results", [])][:5]
    except Exception as e:
        print(f"Movies Error: {e}")
        return ["Unable to fetch movies"]

def get_random_joke():
    try:
        res = requests.get(
            "https://icanhazdadjoke.com/", 
            headers={'Accept': 'application/json'}, 
            timeout=5
        ).json()
        return res["joke"]
    except Exception as e:
        return "Why don't scientists trust atoms? Because they make up everything!"

def get_random_advice():
    try:
        res = requests.get("https://api.adviceslip.com/advice", timeout=5).json()
        return res['slip']['advice']
    except Exception as e:
        return "Stay positive and keep learning!"

def export_conversation():
    """Export conversation history to file"""
    global conversation_history
    
    if not conversation_history:
        return None, "⚠️ No conversation to export"
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"📝 Conversation Export\n")
            f.write(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            for i, (user, bot) in enumerate(conversation_history, 1):
                user_text = user if user else "[Proactive Message]"
                f.write(f"[{i}] 👤 User:\n{user_text}\n\n")
                f.write(f"🤖 {BOTNAME}:\n{bot}\n\n")
                f.write("-"*70 + "\n\n")
        
        return filename, f"✅ Exported {len(conversation_history)} messages to {filename}"
    except Exception as e:
        return None, f"❌ Export failed: {e}"

def export_emotion_history():
    """Export emotion detection history to CSV"""
    global emotion_state, static_emotion_state
    
    # Combine live and static emotion data
    all_emotions = emotion_state.get('history', []) + static_emotion_state.get('history', [])
    
    if not all_emotions:
        return None, "⚠️ No emotion data to export"
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotions_{timestamp}.csv"
        
        df = pd.DataFrame(all_emotions)
        df.to_csv(filename, index=False)
        
        return filename, f"✅ Exported {len(df)} emotion records to {filename}"
    except Exception as e:
        return None, f"❌ Export failed: {e}"

# ==============================================================================
# -------------------- SECTION 5: RAG CHATBOT CORE LOGIC (LlamaIndex) ----------
# ==============================================================================

def create_index(data_dir="./data", persist_dir="./storage"):
    """Create a new LlamaIndex from documents"""
    try:
        if os.path.exists(data_dir) and os.listdir(data_dir):
            print(f"📂 Loading documents from {data_dir}...")
            documents = SimpleDirectoryReader(data_dir).load_data()
            print(f"✅ Loaded {len(documents)} documents")
        else:
            print("⚠️ No documents found in ./data/ directory")
            documents = []
        
        if documents:
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
        else:
            index = VectorStoreIndex([])
        
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"✅ Index created and saved to {persist_dir}")
        
        return index
    
    except Exception as e:
        print(f"Error creating index: {e}")
        return VectorStoreIndex([])

def load_index(persist_dir="./storage"):
    """Load existing LlamaIndex"""
    try:
        if os.path.exists(persist_dir) and os.path.exists(f"{persist_dir}/docstore.json"):
            print(f"📂 Loading existing index from {persist_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("✅ Index loaded successfully")
            return index
        else:
            print("⚠️ No existing index found, creating new one...")
            return create_index(persist_dir=persist_dir)
    except Exception as e:
        print(f"Error loading index: {e}")
        return create_index(persist_dir=persist_dir)

def refresh_query_engine():
    """Refresh the query engine after adding documents"""
    global query_engine, index
    try:
        query_engine = create_query_engine(index)
        print("✅ Query engine refreshed")
    except Exception as e:
        print(f"⚠️ Error refreshing query engine: {e}")

def add_text_to_index(user_text, index, persist_dir="./storage"):
    """Add manual text to the index - FIXED VERSION"""
    if not user_text.strip(): 
        return "⚠️ Please enter valid text"
    
    try:
        # Create document with metadata
        doc = Document(
            text=user_text,
            metadata={
                'source': 'manual_text',
                'timestamp': datetime.now().isoformat(),
                'type': 'personal_info'
            }
        )
        
        # Insert into index
        index.insert(doc)
        
        # Persist immediately
        index.storage_context.persist(persist_dir=persist_dir)
        
        # Refresh query engine
        refresh_query_engine()
        
        # Verify
        docs = get_all_documents(persist_dir)
        
        # Test query immediately
        test_query = "Tell me about the information in the documents"
        test_response = query_engine.query(test_query)
        print(f"\n✅ Test query response: {str(test_response)[:100]}...\n")
        
        return f"""✅ Text added successfully!

📊 **Total documents:** {len(docs)}
🔍 **Test query passed:** Yes

💡 Try asking in Chat:
- "What's my name?"
- "Tell me about myself"
- "What information do you have about me?"
"""
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error adding text: {error_details}")
        return f"❌ Error adding text: {str(e)}"

def add_pdf_to_index(pdf_file, index, persist_dir="./storage"):
    """Process uploaded PDF and add to index - FIXED VERSION"""
    if pdf_file is None:
        return "⚠️ Please upload a PDF file"
    
    try:
        # Gradio 4.x returns file path as string
        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        
        if not os.path.exists(pdf_path):
            return f"❌ File not found: {pdf_path}"
        
        print(f"📄 Processing PDF: {pdf_path}")
        
        # Load PDF using LlamaIndex
        from llama_index.core import SimpleDirectoryReader
        
        # Use input_files parameter with list
        loader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = loader.load_data()
        
        if not documents:
            return "⚠️ Could not extract text from PDF. The file might be empty, corrupted, or scanned (image-based)."
        
        print(f"✅ Extracted {len(documents)} pages")
        
        # Insert each document
        for doc in documents:
            index.insert(doc)
        
        # Persist changes
        index.storage_context.persist(persist_dir=persist_dir)
        
        # Refresh query engine
        refresh_query_engine()
        
        # Verify
        all_docs = get_all_documents(persist_dir)
        
        pdf_name = os.path.basename(pdf_path)
        
        return f"""✅ PDF processed successfully!

📄 **File:** {pdf_name}
📄 **Pages Extracted:** {len(documents)}
📊 **Total Documents:** {len(all_docs)}

💡 You can now ask questions about this PDF in the Chat tab!

**Try asking:**
- "What is this document about?"
- "Summarize the main points"
- "Tell me about [specific topic from PDF]"
"""
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ PDF Error: {error_details}")
        return f"""❌ Error processing PDF: {str(e)}

**Common Issues:**
- PDF is password-protected
- PDF is scanned (image-based, not text-based)
- File is corrupted
- Insufficient permissions

💡 Try:
1. Ensure PDF has selectable text (not scanned images)
2. Remove password protection if any
3. Use a different PDF file

**Technical Details:**
{error_details[:500]}
"""

def get_all_documents(persist_dir="./storage"):
    """Get all documents from the vector database"""
    try:
        docstore_path = os.path.join(persist_dir, "docstore.json")
        
        if not os.path.exists(docstore_path):
            return []
        
        with open(docstore_path, 'r', encoding='utf-8') as f:
            docstore = json.load(f)
        
        documents = []
        doc_dict = docstore.get('docstore/data', {})
        
        for doc_id, doc_data in doc_dict.items():
            doc_info = {
                'id': doc_id,
                'text': doc_data.get('__data__', {}).get('text', '')[:200] + "...",  # Preview
                'metadata': doc_data.get('__data__', {}).get('metadata', {}),
                'full_text': doc_data.get('__data__', {}).get('text', '')
            }
            documents.append(doc_info)
        
        return documents
    
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []

def delete_document_from_index(doc_id, index, persist_dir="./storage"):
    """Delete a specific document from the vector database"""
    try:
        # Delete the document
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
        
        # Persist changes
        index.storage_context.persist(persist_dir=persist_dir)
        
        return f"✅ Document deleted successfully!\n\nDocument ID: {doc_id}"
    
    except Exception as e:
        return f"❌ Error deleting document: {str(e)}"

def clear_all_documents(persist_dir="./storage", data_dir="./data"):
    """Clear all documents from the vector database"""
    try:
        # Remove storage directory
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        # Recreate empty storage
        os.makedirs(persist_dir, exist_ok=True)
        
        # Create new empty index
        new_index = VectorStoreIndex([])
        new_index.storage_context.persist(persist_dir=persist_dir)
        
        return "✅ All documents cleared from knowledge base!", new_index
    
    except Exception as e:
        return f"❌ Error clearing documents: {str(e)}", None

def create_query_engine(index):
    """Create a query engine from the index"""
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact"
    )
    return query_engine

def create_live_search_tool():
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
        def tavily_search(query: str) -> str:
            """Search using Tavily AI for real-time, accurate results"""
            try:
                print(f"🔍 Tavily Search: {query}")
                response = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=True
                )
                
                results = []
                
                if response.get('answer'):
                    results.append(f"📌 AI Summary: {response['answer']}\n")
                
                for idx, result in enumerate(response.get('results', []), 1):
                    results.append(
                        f"\n[Source {idx}]\n"
                        f"Title: {result['title']}\n"
                        f"Content: {result['content']}\n"
                        f"URL: {result['url']}\n"
                        f"Published: {result.get('published_date', 'Recent')}\n"
                    )
                
                final_result = "\n".join(results) if results else "No current results found"
                print(f"✅ Tavily returned {len(response.get('results', []))} results")
                return final_result
                
            except Exception as e:
                print(f"❌ Tavily Search Error: {e}")
                return f"Search error: {str(e)}"
        
        return tavily_search
    
    except Exception as e:
        print(f"❌ Tavily initialization error: {e}")
        return None

def debug_knowledge_base():
    """Debug function to check knowledge base contents"""
    try:
        docs = get_all_documents()
        print("\n" + "="*70)
        print("📚 KNOWLEDGE BASE DEBUG")
        print("="*70)
        print(f"Total documents: {len(docs)}\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Document {i} ---")
            print(f"ID: {doc['id']}")
            print(f"Preview: {doc['text'][:200]}")
            print(f"Full length: {len(doc['full_text'])} characters")
            print("-"*70)
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"Debug error: {e}")

def hybrid_qa(user_input, query_engine, llm, live_search_func):
    """FIXED: Main QA function with improved response validation"""
    global conversation_history
    
    try:
        user_input_lower = user_input.lower()
        
        # Build conversation context
        context_window = 3
        context_string = ""
        
        if conversation_history:
            recent_history = conversation_history[-context_window:]
            context_string = "Previous conversation:\n"
            for user_q, bot_a in recent_history:
                user_text = user_q if user_q else "[Bot was proactive]"
                context_string += f"User: {user_text}\nAssistant: {bot_a}\n\n"
            context_string += f"Current User Query: {user_input}\n"
        else:
            context_string = f"User Query: {user_input}\n"
        
        # Personal query detection
        personal_keywords = [
            'my ', 'i am', 'i work', 'i like', 'my friend', 'my job', 
            'my profession', 'my project', 'my family', 'my name', 
            'about me', 'tell me about my', 'my current', 'my recent',
            'my document', 'my note', 'my data', 'my information',
            'who am i', 'what is my', "what's my", 'tell me my'
        ]
        
        is_personal = any(keyword in user_input_lower for keyword in personal_keywords)
        
        # Follow-up detection
        follow_up_keywords = [
            'for one way', 'one way', 'return', 'what about', 'how about',
            'and that', 'also', 'too', 'as well', 'instead', 'rather',
            'but', 'however', 'or', 'alternatively', 'for ', 'with ',
            'in that case', 'then', 'so', 'more details', 'elaborate',
            'explain', 'tell me more', 'what else', 'anything else'
        ]
        
        is_follow_up = (
            len(conversation_history) > 0 and 
            (
                len(user_input.split()) <= 5 or
                any(keyword in user_input_lower for keyword in follow_up_keywords)
            )
        )
        
        # Time-sensitive detection
        time_sensitive_keywords = [
            'latest', 'today', 'now', 'recent', 'this year', 
            'this month', 'yesterday', 'live', 'currently', 'present',
            'who is the', 'what is the current', 'latest news', 'trending',
            'right now', 'as of', 'update', 'new', 'just', 'today\'s',
            'breaking', 'ongoing', 'happening now', 'captain', 'current team',
            'current', 'present day', 'at the moment', 'price', 'rate',
            'cost', 'flight', 'ticket', 'booking', 'hotel', 'train'
        ]
        
        is_time_sensitive = (
            any(keyword in user_input_lower for keyword in time_sensitive_keywords)
            and not is_personal
        )
        
        # =================================================================
        # FIXED: PERSONAL QUERY HANDLING
        # =================================================================
        if is_personal:
            print(f"🧑 Personal query detected: '{user_input}'")
            print("🔍 Searching knowledge base...")
            
            try:
                query_with_context = context_string if is_follow_up else user_input
                response = query_engine.query(query_with_context)
                
                # Convert to string and check
                response_text = str(response).strip()
                
                print(f"📊 Raw response: '{response_text}'")
                print(f"📏 Response length: {len(response_text)}")
                
                # ✅ FIXED: More lenient validation
                if response_text and len(response_text) > 5:
                    # Check if response is meaningful (not just "Empty Response" or similar)
                    if not any(x in response_text.lower() for x in ['empty response', 'no response', 'none', 'n/a']):
                        print(f"✅ Found relevant information in knowledge base")
                        answer = response_text
                    else:
                        print(f"⚠️ Response seems empty or invalid")
                        answer = "⚠️ I found your information but couldn't extract a clear answer. Let me search more thoroughly..."
                        
                        # Try a more specific query
                        enhanced_query = f"Based on the documents, answer this question: {user_input}"
                        response2 = query_engine.query(enhanced_query)
                        response2_text = str(response2).strip()
                        
                        if response2_text and len(response2_text) > 5:
                            answer = response2_text
                        else:
                            answer = "⚠️ I have your documents but couldn't find specific information to answer that question. Try asking differently or being more specific."
                else:
                    print(f"⚠️ Empty response from knowledge base")
                    answer = "⚠️ I don't have that specific information in your personal knowledge base yet.\n\n💡 You can add it using the '➕ Knowledge' tab."
                    
            except Exception as e:
                print(f"❌ Query error: {e}")
                import traceback
                traceback.print_exc()
                answer = "⚠️ I encountered an error searching your knowledge base. Please try rephrasing your question."
        
        # =================================================================
        # TIME-SENSITIVE OR FOLLOW-UP QUERIES
        # =================================================================
        elif is_time_sensitive or is_follow_up:
            print("⏰ Time-sensitive or follow-up query detected, using Tavily Live Search...")
            
            if live_search_func:
                today = date.today().strftime("%B %d, %Y")
                
                if is_follow_up and conversation_history:
                    last_query, last_answer = conversation_history[-1]
                    enhanced_query = f"Previous context: {last_query}. Follow-up question: {user_input}"
                else:
                    enhanced_query = f"{user_input} as of {today}"
                
                search_results = live_search_func(enhanced_query)
                
                if is_follow_up:
                    prompt = f"""You are continuing a conversation. Use the previous context to answer the follow-up question.

{context_string}

Search Results:
{search_results}

Instructions:
1. Understand that this is a FOLLOW-UP question related to the previous conversation
2. Use the search results to provide specific information
3. Be concise and directly answer what the user is asking for
4. If asking about "one way" or similar, assume they're referring to the previous topic

Provide a helpful, contextual answer:"""
                else:
                    prompt = f"""You are answering a question about CURRENT events as of {today}.

CRITICAL INSTRUCTIONS:
1. If there's an "AI Summary" in the results, prioritize it - it's fact-checked
2. Use the detailed search results to add context and verify information
3. ALWAYS start your answer with "As of {today}, ..."
4. If multiple sources conflict, mention the most recent publication date
5. DO NOT use your training data - ONLY use the search results provided
6. If search results are unclear, say "I couldn't find definitive current information"

Current Date: {today}

Search Results:
{search_results}

Question: {user_input}

Answer (concise, factual, and current):"""
                
                response = llm.complete(prompt)
                answer = str(response)
            else:
                if is_follow_up:
                    prompt = f"""You are continuing a conversation. Answer the follow-up question based on context.

{context_string}

Provide a helpful answer:"""
                    response = llm.complete(prompt)
                    answer = str(response)
                else:
                    answer = "⚠️ Live search is not available. Please check your Tavily API key in the .env file."
        
        # =================================================================
        # GENERAL QUERIES
        # =================================================================
        else:
            try:
                print(f"🔍 General query: '{user_input}'")
                print("🔍 Querying knowledge base first...")
                
                query_text = context_string if is_follow_up else user_input
                response = query_engine.query(query_text)
                response_text = str(response).strip()
                
                print(f"📊 Response: '{response_text[:100]}...'")
                
                # ✅ FIXED: More lenient validation
                if response_text and len(response_text) > 15:
                    print("✅ Found relevant information in knowledge base")
                    answer = response_text
                else:
                    print("⚠️ No relevant docs, using Tavily Live Search...")
                    
                    if live_search_func:
                        if is_follow_up and conversation_history:
                            last_query, _ = conversation_history[-1]
                            search_query = f"{last_query} {user_input}"
                        else:
                            search_query = user_input
                        
                        search_results = live_search_func(search_query)
                        
                        if is_follow_up:
                            prompt = f"""Continue the conversation by answering this follow-up question.

{context_string}

Search Results:
{search_results}

Provide a concise, contextual answer:"""
                        else:
                            prompt = f"""Answer the following question based on the provided search results:

Search Results:
{search_results}

Question: {user_input}

Answer (be concise and informative):"""
                        
                        response = llm.complete(prompt)
                        answer = str(response)
                    else:
                        if is_follow_up:
                            prompt = f"""Continue the conversation:

{context_string}

Answer:"""
                            response = llm.complete(prompt)
                            answer = str(response)
                        else:
                            response = llm.complete(user_input)
                            answer = str(response)
            
            except Exception as e:
                print(f"❌ Query error: {e}, falling back to LLM")
                if is_follow_up:
                    prompt = f"""Continue the conversation:

{context_string}

Answer:"""
                    response = llm.complete(prompt)
                    answer = str(response)
                else:
                    response = llm.complete(user_input)
                    answer = str(response)

        # Save to history
        conversation_history.append((user_input, answer))
        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]
        
        return answer
    
    except Exception as e:
        print(f"❌ Error in hybrid_qa: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I encountered an error: {str(e)}"

# ==============================================================================
# -------------------- SECTION 6: COMMAND HANDLER ------------------------------
# ==============================================================================

def handle_text_query(query):
    """Handles voice assistant commands from text input"""
    global whatsapp_state, email_state
    
    if not query.strip(): 
        return None

    query_lower = query.lower()
    
    # WhatsApp Conversation Flow Handler
    if whatsapp_state['active']:
        if 'cancel' in query_lower or 'stop' in query_lower or 'exit' in query_lower:
            whatsapp_state['active'] = False
            whatsapp_state['number'] = None
            whatsapp_state['stage'] = None
            return "❌ WhatsApp message cancelled."
        
        if whatsapp_state['stage'] == 'waiting_for_number':
            number = query.strip()
            if number.isdigit() and len(number) == 10:
                whatsapp_state['number'] = number
                whatsapp_state['stage'] = 'waiting_for_message'
                return f"✅ Number saved: +91{number}\n\n💬 Now, what message do you want to send?\n\n💡 Type 'cancel' to abort"
            else:
                return "⚠️ Invalid phone number. Please enter a 10-digit number.\n💡 Example: 9876543210\n\n🛑 Type 'cancel' to abort"
        
        elif whatsapp_state['stage'] == 'waiting_for_message':
            message = query.strip()
            if message:
                number = whatsapp_state['number']
                success, msg = send_whatsapp_message(number, message)
                response = f"✅ WhatsApp message sent!\n\n📱 To: +91{number}\n💬 Message: {message}" if success else msg
                whatsapp_state['active'] = False
                whatsapp_state['number'] = None
                whatsapp_state['stage'] = None
                return response
            else:
                return "⚠️ Message cannot be empty. Please type your message.\n\n🛑 Type 'cancel' to abort"
    
    # Email Conversation Flow Handler
    if email_state['active']:
        if 'cancel' in query_lower or 'stop' in query_lower or 'exit' in query_lower:
            email_state['active'] = False
            email_state['email'] = None
            email_state['subject'] = None
            email_state['stage'] = None
            return "❌ Email sending cancelled."
        
        if email_state['stage'] == 'waiting_for_email':
            email_address = query.strip()
            if '@' in email_address and '.' in email_address:
                email_state['email'] = email_address
                email_state['stage'] = 'waiting_for_subject'
                return f"✅ Email saved: {email_address}\n\n📝 Now, what's the subject?\n\n💡 Type 'cancel' to abort"
            else:
                return "⚠️ Invalid email address. Please enter a valid email.\n💡 Example: john@gmail.com\n\n🛑 Type 'cancel' to abort"
        
        elif email_state['stage'] == 'waiting_for_subject':
            subject = query.strip()
            if subject:
                email_state['subject'] = subject
                email_state['stage'] = 'waiting_for_message'
                return f"✅ Subject saved: {subject}\n\n💬 Now, what's the message?\n\n💡 Type 'cancel' to abort"
            else:
                return "⚠️ Subject cannot be empty. Please type the subject.\n\n🛑 Type 'cancel' to abort"
        
        elif email_state['stage'] == 'waiting_for_message':
            message = query.strip()
            if message:
                email_address = email_state['email']
                subject = email_state['subject']
                success, msg = send_email(email_address, subject, message)
                response = f"✅ Email sent!\n\n📧 To: {email_address}\n📝 Subject: {subject}\n💬 Message: {message}" if success else msg
                email_state['active'] = False
                email_state['email'] = None
                email_state['subject'] = None
                email_state['stage'] = None
                return response
            else:
                return "⚠️ Message cannot be empty. Please type your message.\n\n🛑 Type 'cancel' to abort"
    
    # Shopping website detection
    shopping_websites = ['ajio', 'amazon', 'flipkart', 'myntra', 'meesho', 'snapdeal', 
                        'nykaa', 'ebay', 'shopclues', 'paytmmall', 'tatacliq', 'jiomart', 
                        'bigbasket', 'swiggy', 'zomato']
    
    for website in shopping_websites:
        if website in query_lower:
            if ' on ' + website in query_lower:
                parts = query_lower.split(' on ' + website)
                search_term = parts[0].strip()
                for prefix in ['search for ', 'find ', 'look for ', 'show me ', 'get me ', 'search ']:
                    search_term = search_term.replace(prefix, '').strip()
                
                if search_term:
                    success = search_on_website(search_term, website)
                    if success:
                        return f"🛍️ Opening {website.title()} and searching for:\n'{search_term}'"
                    else:
                        return f"❌ Could not open {website.title()}"
            
            elif 'open ' + website in query_lower:
                success = open_website(website)
                if success:
                    return f"🌐 Opening {website.title()}"
                else:
                    return f"❌ Could not open {website.title()}"
    
    if 'open ' in query_lower and not any(x in query_lower for x in ['notepad', 'discord', 'cmd', 'camera', 'calculator']):
        website = query_lower.replace('open ', '').strip()
        website = website.replace('website', '').strip()
        
        if website:
            success = open_website(website)
            if success:
                return f"🌐 Opening {website.title()}"
            else:
                return f"⚠️ Trying to open {website}..."
    
    # Command Detection
    if 'whatsapp' in query_lower or 'send message' in query_lower:
        whatsapp_state['active'] = True
        whatsapp_state['stage'] = 'waiting_for_number'
        return "📱 **WhatsApp Message Service**\n\nPlease enter the 10-digit phone number:\n💡 Example: 9876543210\n\n🛑 Type 'cancel' anytime to abort"
    
    if 'email' in query_lower or 'send email' in query_lower or 'mail' in query_lower:
        email_state['active'] = True
        email_state['stage'] = 'waiting_for_email'
        return "📧 **Email Service**\n\nPlease enter the recipient's email address:\n💡 Example: john@gmail.com\n\n🛑 Type 'cancel' anytime to abort"
    
    if 'open notepad' in query_lower:
        open_notepad()
        return "✅ Opening Notepad"
    
    elif 'open discord' in query_lower:
        open_discord()
        return "✅ Opening Discord"
    
    elif 'open command prompt' in query_lower or 'open cmd' in query_lower:
        open_cmd()
        return "✅ Opening Command Prompt"
    
    elif 'open camera' in query_lower:
        open_camera()
        return "✅ Opening Camera"
    
    elif 'open calculator' in query_lower:
        open_calculator()
        return "✅ Opening Calculator"
    
    elif 'ip address' in query_lower or 'my ip' in query_lower:
        ip_address = find_my_ip()
        return f'🌐 Your IP Address is: {ip_address}'
    
    elif 'wikipedia' in query_lower:
        search_query = query_lower.replace('wikipedia', '').replace('search', '').strip()
        if search_query:
            results = search_on_wikipedia(search_query)
            return f"📚 According to Wikipedia:\n{results}"
        else:
            return "⚠️ Please specify what to search. Example: 'wikipedia artificial intelligence'"
    
    elif 'on youtube' in query_lower or ('play' in query_lower and 'video' in query_lower):
        video = query_lower.replace('youtube', '').replace('play', '').replace('on', '').replace('video', '').strip()
        if video:
            play_on_youtube(video)
            return f"🎥 Playing '{video}' on YouTube"
        else:
            return "⚠️ Please specify what to play. Example: 'play music on youtube'"
    
    elif 'search on google' in query_lower or 'google search' in query_lower:
        search_query = query_lower.replace('search on google', '').replace('google search', '').replace('google', '').strip()
        if search_query:
            search_on_google(search_query)
            return f"🔍 Searching for '{search_query}' on Google"
        else:
            return "⚠️ Please specify what to search. Example: 'search on google python tutorials'"
    
    elif 'joke' in query_lower:
        joke = get_random_joke()
        return f"😂 {joke}"
    
    elif 'advice' in query_lower:
        advice = get_random_advice()
        return f"💡 {advice}"
    
    elif 'trending movies' in query_lower or 'latest movies' in query_lower:
        movies = get_trending_movies()
        return "🎬 **Trending Movies:**\n" + "\n".join([f"• {m}" for m in movies])
    
    elif 'news' in query_lower or 'headlines' in query_lower:
        headlines = get_latest_news()
        return "📰 **Top Headlines:**\n" + "\n".join([f"• {h}" for h in headlines])
    
    elif 'weather' in query_lower:
        city = None
        
        if ' in ' in query_lower:
            parts = query_lower.split(' in ')
            if len(parts) > 1:
                city_raw = parts[1].strip()
                for word in ['what', 'is', 'the', 'how', 'tell', 'me', 'about', '?']:
                    city_raw = city_raw.replace(word, '').strip()
                city = city_raw.title()
        
        elif ' at ' in query_lower:
            parts = query_lower.split(' at ')
            if len(parts) > 1:
                city_raw = parts[1].strip()
                for word in ['what', 'is', 'the', '?']:
                    city_raw = city_raw.replace(word, '').strip()
                city = city_raw.title()
        
        elif ' for ' in query_lower:
            parts = query_lower.split(' for ')
            if len(parts) > 1:
                city_raw = parts[1].strip()
                for word in ['what', 'is', 'the', '?']:
                    city_raw = city_raw.replace(word, '').strip()
                city = city_raw.title()
        
        if not city:
            ip_address = find_my_ip()
            try:
                city = requests.get(f"https://ipapi.co/{ip_address}/city/", timeout=5).text
            except:
                city = "Unknown"
        
        weather, temp, feel = get_weather_report(city)
        return f"🌤️ **Weather in {city}:**\n• Condition: {weather}\n• Temperature: {temp}\n• Feels Like: {feel}"
    
    else:
        return None

# ==============================================================================
# -------------------- SECTION 7: TTS & STT FOR RAG CHATBOT --------------------
# ==============================================================================

def text_to_speech_chatbot(user_input):
    """Takes text input, gets answer from bot, converts to speech"""
    if not user_input.strip():
        return None
    
    try:
        command_result = handle_text_query(user_input)
        
        if command_result:
            answer = command_result
        else:
            answer = hybrid_qa(user_input, query_engine, llm, live_search_func)
        
        tts = gTTS(text=answer, lang='en', slow=False)
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(output_file.name)
        
        return output_file.name
    
    except Exception as e:
        print(f"TTS Error: {e}")
        try:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            tts = gTTS(text=error_msg, lang='en', slow=False)
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(output_file.name)
            return output_file.name
        except:
            return None

def speech_to_text_chatbot(audio_file):
    """Takes audio input, transcribes it, gets text answer"""
    if not audio_file:
        return "⚠️ Please provide an audio input"
    
    try:
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en"
            )
        
        transcribed_text = transcription.text
        print(f"🎤 Transcribed: {transcribed_text}")
        
        command_result = handle_text_query(transcribed_text)
        
        if command_result:
            answer = command_result
        else:
            answer = hybrid_qa(transcribed_text, query_engine, llm, live_search_func)
        
        return f"**You said:** {transcribed_text}\n\n**Bot response:** {answer}"
    
    except Exception as e:
        print(f"STT Error: {e}")
        return f"❌ Speech recognition error: {str(e)}"

def speech_to_speech_chatbot(audio_file):
    """Takes audio, transcribes, gets answer, converts to speech"""
    if not audio_file:
        return None
    
    try:
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en"
            )
        
        transcribed_text = transcription.text
        print(f"🎤 Transcribed: {transcribed_text}")
        
        command_result = handle_text_query(transcribed_text)
        
        if command_result:
            answer = command_result
        else:
            answer = hybrid_qa(transcribed_text, query_engine, llm, live_search_func)
        
        tts = gTTS(text=answer, lang='en', slow=False)
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(output_file.name)
        
        return output_file.name
    
    except Exception as e:
        print(f"STS Error: {e}")
        try:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            tts = gTTS(text=error_msg, lang='en', slow=False)
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(output_file.name)
            return output_file.name
        except:
            return None

# ==============================================================================
# -------- SECTION 8: CHATBOT & EMOTION FUNCTIONS ------------------------------
# ==============================================================================

def chatbot_response(message, history):
    """Main chatbot response function"""
    if not message.strip(): 
        return "⚠️ Please provide a valid input"
    
    command_result = handle_text_query(message)
    
    if command_result:
        return command_result
    
    try:
        return hybrid_qa(message, query_engine, llm, live_search_func)
    except Exception as e:
        return f"❌ Error: {str(e)}"

def start_emotion_camera():
    """Start real-time emotion detection"""
    global emotion_detector, emotion_state
    
    try:
        if emotion_detector.start_camera():
            emotion_state['is_camera_running'] = True
            return "✅ Camera started! Detecting emotions in real-time..."
        else:
            return "❌ Failed to start camera. Check if webcam is available."
    except Exception as e:
        return f"❌ Error: {e}"

def stop_emotion_camera():
    """Stop real-time emotion detection"""
    global emotion_detector, emotion_state
    
    try:
        emotion_detector.stop_camera()
        emotion_state['is_camera_running'] = False
        return "🛑 Camera stopped"
    except Exception as e:
        return f"❌ Error: {e}"

def get_emotion_frame():
    """Get current emotion detection frame"""
    global emotion_detector
    
    try:
        frame = emotion_detector.get_latest_frame()
        if frame is not None:
            # Convert BGR to RGB for Gradio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    except:
        return None

def get_current_emotion_status():
    """Get current emotion status and set chat trigger if applicable"""
    global emotion_detector, emotion_state, emotion_chat_trigger
    
    try:
        state = emotion_detector.get_current_state()
        
        emoji_map = {
            'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
            'Surprise': '😮', 'Fear': '😨', 'Disgust': '🤢', 'Neutral': '😐'
        }
        
        emotion = state['emotion']
        confidence = state['confidence']
        face_detected = state['face_detected']
        
        # Update global state and add to history
        emotion_state['current_emotion'] = emotion
        emotion_state['confidence'] = confidence
        
        # Add to history every 2 seconds to avoid duplicates
        if not emotion_state['history'] or \
           (datetime.now() - emotion_state['history'][-1].get('timestamp', datetime.min)).seconds >= 2:
            emotion_state['history'].append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Keep last 50 records
            if len(emotion_state['history']) > 50:
                emotion_state['history'] = emotion_state['history'][-50:]
        
        emoji = emoji_map.get(emotion, '😐')
        status = f"# {emoji} **{emotion}**\n\n**Confidence:** {confidence:.1%}"
        
        if not face_detected:
            status += "\n\n⚠️ No face detected"
        else:
            # NEW: Emotion Trigger Logic
            if (not emotion_chat_trigger['triggered'] and 
                face_detected and 
                confidence > EMOTION_TRIGGER_THRESHOLD and 
                emotion != 'Neutral'):
                
                current_time = time.time()
                if (current_time - emotion_chat_trigger['last_consumed_timestamp']) > EMOTION_TRIGGER_COOLDOWN:
                    emotion_chat_trigger['triggered'] = True
                    emotion_chat_trigger['emotion'] = emotion
                    print(f"🔥 Emotion trigger set: {emotion}")
            
            # Display the message if a trigger is pending
            if emotion_chat_trigger['triggered']:
                status += "\n\n**✨ I have a question for you in the Chat tab!**"
        
        return status
    except:
        return "⚠️ Camera not running"


def update_realtime_analytics():
    """Update real-time emotion analytics"""
    global emotion_state
    
    if not emotion_state['history']:
        empty_df = pd.DataFrame({"Emotion": ["No Data"], "Count": [0]})
        return empty_df, "📊 No data yet.\n\nStart camera to collect emotion data!"
    
    # Count emotions
    emotion_counts = {}
    total = len(emotion_state['history'])
    
    for record in emotion_state['history']:
        emotion = record['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Emotion': list(emotion_counts.keys()),
        'Count': list(emotion_counts.values())
    }).sort_values('Count', ascending=False)
    
    # Create stats text
    emoji_map = {
        'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
        'Surprise': '😮', 'Fear': '😨', 'Disgust': '🤢', 'Neutral': '😐'
    }
    
    stats_text = f"# 📊 **Emotion Analytics**\n\n**Total Detections:** {total}\n\n## 📈 Distribution:\n\n"
    
    for _, row in df.iterrows():
        emotion = row['Emotion']
        count = row['Count']
        percentage = (count / total) * 100
        bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
        emoji = emoji_map.get(emotion, '😐')
        stats_text += f"{emoji} **{emotion:<10}** `{bar}` {count:>2} ({percentage:.1f}%)\n"
    
    # Most common
    most_common = df.iloc[0]
    stats_text += f"\n---\n\n🏆 **Most Common:** {most_common['Emotion']} ({most_common['Count']} times)"
    
    # Average confidence
    avg_conf = sum(r['confidence'] for r in emotion_state['history']) / total
    stats_text += f"\n\n📈 **Avg Confidence:** {avg_conf:.1%}"
    
    # Recent 5
    if len(emotion_state['history']) >= 5:
        recent = [r['emotion'] for r in emotion_state['history'][-5:]]
        stats_text += f"\n\n## 🔄 **Recent (last 5):**\n\n"
        for i, e in enumerate(recent, 1):
            emoji = emoji_map.get(e, '😐')
            stats_text += f"{i}. {emoji} {e}\n"
    
    return df, stats_text

def detect_emotion_from_static_image(image):
    """Detect emotion from uploaded static image"""
    global static_emotion_detector
    return static_emotion_detector.detect_emotion_from_image(image)

# ==============================================================================
# -------------------- SECTION 9: VECTOR DB MANAGEMENT -------------------------
# ==============================================================================

def list_documents():
    """List all documents in the vector database"""
    try:
        docs = get_all_documents()
        
        if not docs:
            return pd.DataFrame({"Message": ["No documents in knowledge base"]}), "📂 Knowledge base is empty"
        
        # Create DataFrame for display
        doc_list = []
        for doc in docs:
            preview = doc['text']
            if len(preview) > 100:
                preview = preview[:100] + "..."
            
            doc_list.append({
                'ID': doc['id'][:8] + "...",  # Short ID
                'Preview': preview,
                'Full_ID': doc['id']  # Hidden full ID
            })
        
        df = pd.DataFrame(doc_list)
        
        # Create info text
        info_text = f"📚 **Total Documents:** {len(docs)}\n\n"
        info_text += "💡 Copy the Full ID to delete a specific document"
        
        return df[['ID', 'Preview']], info_text
        
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]}), f"❌ Error: {str(e)}"

def delete_specific_document(doc_id):
    """Delete a specific document by ID"""
    global index, query_engine
    
    if not doc_id or not doc_id.strip():
        return "⚠️ Please enter a document ID", list_documents()[0], list_documents()[1]
    
    try:
        result = delete_document_from_index(doc_id.strip(), index)
        
        # Refresh query engine
        refresh_query_engine()
        
        # Refresh the document list
        df, info = list_documents()
        
        return result, df, info
        
    except Exception as e:
        df, info = list_documents()
        return f"❌ Error: {str(e)}", df, info

def clear_all_knowledge():
    """Clear entire knowledge base"""
    global index, query_engine
    
    try:
        result, new_index = clear_all_documents()
        
        if new_index:
            index = new_index
            query_engine = create_query_engine(index)
        
        # Refresh the document list
        df, info = list_documents()
        
        return result, df, info
        
    except Exception as e:
        df, info = list_documents()
        return f"❌ Error: {str(e)}", df, info

# ==============================================================================
# -------------------- SECTION 9.5: CROSS-TAB INTERACTION ----------------------
# ==============================================================================

def check_emotion_trigger_and_update_chat(chat_history):
    """Checks for an emotion trigger and adds a proactive message to the chat."""
    global emotion_chat_trigger, conversation_history

    if emotion_chat_trigger.get('triggered'):
        emotion = emotion_chat_trigger.get('emotion')
        
        emotion_prompts = {
            'Sad': f"I noticed you seem a bit sad. Is everything okay? I'm here to listen if you'd like to talk about what happened today.",
            'Angry': f"You seem angry. If you'd like to talk about what's bothering you, I'm here to listen.",
            'Happy': f"You look really happy! That's wonderful to see. Did something great happen today?",
            'Surprise': f"You seem surprised! I hope it was a pleasant one. Did something unexpected happen today?",
            'Fear': f"I detected a hint of fear. I hope everything is alright. Please know that I'm here if you need to talk.",
            'Disgust': f"I sensed you might be feeling disgusted. I hope you're okay. What's on your mind?"
        }
        
        bot_message = emotion_prompts.get(emotion, f"I sensed you might be feeling {emotion.lower()}. What's on your mind today?")
        
        # Add to Gradio's chat history
        if chat_history is None:
            chat_history = []
        chat_history.append([None, bot_message])
        
        # Add to the RAG bot's context history
        conversation_history.append(("[Proactive based on emotion]", bot_message))
        if len(conversation_history) > MAX_HISTORY:
            conversation_history.pop(0)

        # Consume the trigger and start cooldown timer
        emotion_chat_trigger['triggered'] = False
        emotion_chat_trigger['emotion'] = None
        emotion_chat_trigger['last_consumed_timestamp'] = time.time()
        
        print(f"✅ Chat updated with proactive question for emotion: {emotion}. Cooldown started.")
        
    return chat_history

# Cleanup function for app exit
def cleanup():
    """Cleanup resources on exit"""
    try:
        global emotion_detector
        if emotion_detector and emotion_detector.is_running:
            emotion_detector.stop_camera()
            print("🧹 Cleaned up camera resources")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

# Register cleanup
atexit.register(cleanup)

# ==============================================================================
# -------------------- SECTION 10: INITIALIZATION & GRADIO UI ------------------
# ==============================================================================

print("🚀 Initializing Personal RAG Chatbot with LlamaIndex & Tavily AI...")
print("="*70)

try:
    llm, embed_model = initialize_llama_index()
    print("✅ LlamaIndex initialized successfully")
except Exception as e:
    print(f"❌ LlamaIndex initialization failed: {e}")
    exit(1)

# Create necessary directories
os.makedirs("./data", exist_ok=True)
os.makedirs("./storage", exist_ok=True)

# Load or create index
index = load_index(persist_dir="./storage")
query_engine = create_query_engine(index)

# Initialize Tavily search
live_search_func = create_live_search_tool()

if live_search_func:
    print("✅ Tavily AI Search initialized successfully")
else:
    print("⚠️ Warning: Tavily AI Search failed to initialize.")

print("✅ System Ready!")
print("="*70)

# Debug knowledge base on startup
print("\n🔍 Checking knowledge base on startup...")
debug_knowledge_base()

print("\n🎭 **REAL-TIME EMOTION DETECTION:**")
print("   • TensorFlow + OpenCV")
print("   • 7 emotions with live tracking")
print("   • Static image emotion detection")
print("   • Export emotion data to CSV")
print("\n🛍️ **SHOPPING COMMANDS:**")
print("   • 'laptop on amazon'")
print("   • 'red dress on myntra'")
print("\n📊 **NEW FEATURES:**")
print("   • Proactive chat based on detected emotion!")
print("   • Export conversation history")
print("   • Export emotion analytics")
print("   • Enhanced error handling")
print("   • Auto cleanup on exit")
print("="*70 + "\n")

# ==============================================================================
# ------------------ GRADIO UI (COMPLETE WITH ALL FEATURES) --------------------
# ==============================================================================

custom_css = """
.emotion-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}
.stats-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="GKPS AI") as app:
    
    gr.Markdown("""
    # 🤖 GKPS AI 
    ### 🎭 Emotion Detection | 📸 Static Analysis | 🛍️ Smart Shopping | 🧠 RAG | 🎙️ Voice | 🔍 Live Search | 🗑️ Vector DB | 📊 Export
    """)
    
    # ==================== TAB 1: RAG CHATBOT ====================
    with gr.Tab("💬 Chat") as chat_tab:
        gr.Markdown("# 💬 Intelligent Assistant")
        
        chatbot_display = gr.Chatbot(height=400, label="Conversation")
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask anything...",
                scale=9,
                show_label=False
            )
            send_btn = gr.Button("Send 📤", scale=1, variant="primary")
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat")
            clear_ctx_btn = gr.Button("🔄 Reset Context")
            export_conv_btn = gr.Button("💾 Export Chat", variant="secondary")
        
        export_status = gr.Textbox(label="Export Status", lines=2, visible=False)
        
        def clear_context():
            global conversation_history
            conversation_history = []
            return "✅ Context cleared"
        
        clear_ctx_btn.click(clear_context, None, msg)
        
        def export_chat():
            file, status = export_conversation()
            if file:
                return status, gr.update(visible=True)
            return status, gr.update(visible=True)
        
        export_conv_btn.click(export_chat, None, [export_status, export_status])
        
        def user_msg(user, hist):
            return "", hist + [[user, None]]
        
        def bot_msg(hist):
            user = hist[-1][0]
            bot = chatbot_response(user, hist)
            hist[-1][1] = bot
            return hist
        
        msg.submit(user_msg, [msg, chatbot_display], [msg, chatbot_display]).then(
            bot_msg, chatbot_display, chatbot_display
        )
        send_btn.click(user_msg, [msg, chatbot_display], [msg, chatbot_display]).then(
            bot_msg, chatbot_display, chatbot_display
        )
        clear_btn.click(lambda: None, None, chatbot_display)
        
        gr.Markdown("---")
        gr.Markdown("## 🎙️ Voice Mode")
        
        voice_mode = gr.Dropdown(
            ["📢 TTS", "🎤 STT", "🎙️ STS"],
            value="📢 TTS",
            label="Mode"
        )
        
        with gr.Row():
            with gr.Column(visible=True) as tts_col:
                gr.Markdown("### 📢 Text-to-Speech")
                tts_in = gr.Textbox(label="Input", lines=2)
                tts_btn = gr.Button("🔊 Generate", variant="primary")
                tts_out = gr.Audio(label="Output", autoplay=False)
            
            with gr.Column(visible=False) as stt_col:
                gr.Markdown("### 🎤 Speech-to-Text")
                stt_in = gr.Audio(
                    label="Record",
                    sources=["microphone", "upload"],
                    type="filepath"
                )
                stt_btn = gr.Button("📝 Transcribe", variant="primary")
                stt_out = gr.Textbox(label="Output", lines=5)
            
            with gr.Column(visible=False) as sts_col:
                gr.Markdown("### 🎙️ Speech-to-Speech")
                sts_in = gr.Audio(
                    label="Record",
                    sources=["microphone", "upload"],
                    type="filepath"
                )
                sts_btn = gr.Button("🔊 Process", variant="primary")
                sts_out = gr.Audio(label="Output", autoplay=False)
        
        def toggle_voice(mode):
            return (
                gr.update(visible=mode=="📢 TTS"),
                gr.update(visible=mode=="🎤 STT"),
                gr.update(visible=mode=="🎙️ STS")
            )
        
        voice_mode.change(toggle_voice, voice_mode, [tts_col, stt_col, sts_col])
        
        tts_btn.click(text_to_speech_chatbot, tts_in, tts_out)
        stt_btn.click(speech_to_text_chatbot, stt_in, stt_out)
        sts_btn.click(speech_to_speech_chatbot, sts_in, sts_out)
    
    # ==================== TAB 2: REAL-TIME EMOTION DETECTION ====================
    with gr.Tab("🎭 Emotion"):
        gr.Markdown("# 🎭 Emotion Detection System")
        gr.Markdown("**Real-time webcam & static image emotion recognition with analytics**")
        
        with gr.Tabs():
            # Real-time Camera Tab
            with gr.Tab("📹 Live Camera"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## 📹 Live Feed")
                        emotion_video = gr.Image(label="Webcam Feed", type="pil", height=400)
                        
                        with gr.Row():
                            start_camera_btn = gr.Button("▶️ Start Camera", variant="primary", size="lg")
                            stop_camera_btn = gr.Button("⏹️ Stop Camera", variant="stop", size="lg")
                        
                        camera_status = gr.Textbox(label="Status", value="⚪ Camera stopped", interactive=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("## 😊 Current Emotion", elem_classes="emotion-card")
                        emotion_status = gr.Markdown("# 😐 **Neutral**\n\n**Confidence:** 0%\n\n⚠️ Camera not running")
                        
                        gr.Markdown("### ⚙️ Settings")
                        auto_refresh = gr.Checkbox(label="Auto-refresh (1 sec)", value=False)
                        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                
                # Camera controls
                start_camera_btn.click(fn=start_emotion_camera, outputs=camera_status)
                stop_camera_btn.click(fn=stop_emotion_camera, outputs=camera_status)
                
                # Manual refresh
                refresh_btn.click(fn=get_current_emotion_status, outputs=emotion_status)
                refresh_btn.click(fn=get_emotion_frame, outputs=emotion_video)
                
                # Auto-refresh every 1 second
                timer = gr.Timer(value=1.0, active=False)
                auto_refresh.change(lambda x: gr.Timer(active=x), auto_refresh, timer)
                timer.tick(fn=get_current_emotion_status, outputs=emotion_status)
                timer.tick(fn=get_emotion_frame, outputs=emotion_video)
                
                gr.Markdown("---")
                gr.Markdown("## 📊 Real-Time Analytics Dashboard")
                gr.Markdown("**Track emotion trends over time** (Last 50 detections)")
                
                with gr.Row():
                    with gr.Column():
                        analytics_plot = gr.BarPlot(
                            value=pd.DataFrame({"Emotion": ["No Data"], "Count": [0]}),
                            x="Emotion",
                            y="Count",
                            title="Emotion Distribution",
                            width=500,
                            height=350,
                            color="Emotion"
                        )
                    
                    with gr.Column():
                        analytics_text = gr.Markdown(
                            "📊 No data yet.\n\nStart camera to collect emotion data!",
                            elem_classes="stats-card"
                        )
                
                with gr.Row():
                    update_analytics_btn = gr.Button("📈 Update Analytics", variant="primary", size="lg")
                    export_emotion_btn = gr.Button("💾 Export Data (CSV)", variant="secondary", size="lg")
                
                emotion_export_status = gr.Textbox(label="Export Status", lines=2, visible=False)
                
                update_analytics_btn.click(fn=update_realtime_analytics, outputs=[analytics_plot, analytics_text])
                
                def export_emotions():
                    file, status = export_emotion_history()
                    if file:
                        return status, gr.update(visible=True)
                    return status, gr.update(visible=True)
                
                export_emotion_btn.click(export_emotions, None, [emotion_export_status, emotion_export_status])
            
            # Static Image Tab
            with gr.Tab("📸 Static Image"):
                gr.Markdown("## 📸 Upload Image for Emotion Detection")
                gr.Markdown("**Upload any image with a face to detect emotions with detailed analysis**")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        static_image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=400
                        )
                        
                        detect_static_btn = gr.Button(
                            "🔍 Detect Emotion",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        static_image_output = gr.Image(
                            label="Annotated Result",
                            type="pil",
                            height=400
                        )
                        
                        static_emotion_status = gr.Markdown(
                            "📸 Upload an image and click 'Detect Emotion'",
                            elem_classes="stats-card"
                        )
                
                # Static image detection
                detect_static_btn.click(
                    fn=detect_emotion_from_static_image,
                    inputs=static_image_input,
                    outputs=[static_image_output, static_emotion_status]
                )
                
                gr.Markdown("---")
                gr.Markdown("""
                ### 💡 Tips for Best Results:
                - ✅ Use clear, well-lit images
                - ✅ Face should be clearly visible
                - ✅ Front-facing photos work best
                - ✅ Supports JPG, PNG formats
                - ✅ High-quality images recommended
                """)
    
    # ==================== TAB 3: ADD KNOWLEDGE ====================
    with gr.Tab("➕ Knowledge"):
        gr.Markdown("# 📚 Add to Knowledge Base")
        gr.Markdown("**Build your personal AI knowledge base with notes and documents**")
        
        with gr.Tabs():
            with gr.Tab("📝 Text"):
                gr.Markdown("""
                ### Add Personal Notes & Information
                Add any text information you want the AI to remember and reference later.
                
                **Examples:**
                - Personal facts: "My name is John. I work as a software engineer at Google."
                - Project notes: "I'm working on a machine learning project using TensorFlow."
                - Important dates: "My anniversary is on June 15th."
                - Preferences: "I prefer Python over JavaScript for backend development."
                """)
                
                text_input = gr.Textbox(
                    lines=10,
                    placeholder="Enter text to add to knowledge base...\n\nExample:\nMy name is John. I work as a Data Scientist at ABC Corp.\nI specialize in NLP and computer vision.\nMy current project involves building a chatbot.",
                    label="Your Text"
                )
                
                text_btn = gr.Button("➕ Add Text to Knowledge Base", variant="primary", size="lg")
                text_status = gr.Textbox(label="Status", lines=4, interactive=False)
                
                # Clear input after successful add
                def add_text_and_clear(text):
                    result = add_text_to_index(text, index)
                    if "✅" in result:
                        return result, ""  # Clear input on success
                    return result, text  # Keep input on error
                
                text_btn.click(
                    add_text_and_clear,
                    inputs=text_input,
                    outputs=[text_status, text_input]
                )
                
                gr.Markdown("""
                ---
                ### 💡 Tips for Adding Text:
                - Be specific and clear
                - Add context to your information
                - You can add multiple facts at once
                - Use natural language
                """)
            
            with gr.Tab("📄 PDF"):
                gr.Markdown("""
                ### Upload PDF Documents
                Upload PDF files to add them to your knowledge base.
                
                **Supported:**
                - ✅ Text-based PDFs
                - ✅ Research papers
                - ✅ Articles
                - ✅ Books
                - ✅ Reports
                
                **Not Supported:**
                - ❌ Scanned PDFs (images only)
                - ❌ Password-protected PDFs
                """)
                
                pdf_input = gr.File(
                    label="📤 Choose PDF File",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                pdf_btn = gr.Button("📥 Process PDF", variant="primary", size="lg")
                pdf_status = gr.Textbox(label="Status", lines=8, interactive=False)
                
                # Clear input after successful add
                def add_pdf_and_clear(pdf):
                    if pdf is None:
                        return "⚠️ Please select a PDF file", None
                    result = add_pdf_to_index(pdf, index)
                    if "✅" in result:
                        return result, None  # Clear input on success
                    return result, pdf  # Keep input on error
                
                pdf_btn.click(
                    add_pdf_and_clear,
                    inputs=pdf_input,
                    outputs=[pdf_status, pdf_input]
                )
                
                gr.Markdown("""
                ---
                ### 📋 How to Use:
                1. Click "Choose PDF File"
                2. Select your PDF
                3. Click "Process PDF"
                4. Wait for processing (may take a few seconds)
                5. Check the status message
                
                ### ⚠️ Troubleshooting:
                - **No text extracted?** PDF might be scanned. Use OCR software first.
                - **Error message?** Check file isn't password-protected.
                - **Slow processing?** Large PDFs take longer. Be patient.
                """)
            
            with gr.Tab("🔍 Verify"):
                gr.Markdown("""
                ### 🔍 Verify Knowledge Base
                Check if your documents were added successfully.
                """)
                
                verify_btn = gr.Button("🔄 Check Knowledge Base", variant="secondary", size="lg")
                verify_output = gr.Textbox(label="Knowledge Base Status", lines=10, interactive=False)
                
                def verify_knowledge_base():
                    try:
                        docs = get_all_documents()
                        
                        if not docs:
                            return "📂 Knowledge base is empty\n\n💡 Add some text or PDF documents to get started!"
                        
                        result = f"✅ Knowledge Base Status:\n\n"
                        result += f"📊 **Total Documents:** {len(docs)}\n\n"
                        result += "="*50 + "\n\n"
                        
                        for i, doc in enumerate(docs[:10], 1):  # Show first 10
                            preview = doc['text'][:150].replace('\n', ' ')
                            result += f"**Document {i}:**\n"
                            result += f"ID: {doc['id'][:20]}...\n"
                            result += f"Preview: {preview}...\n\n"
                            result += "-"*50 + "\n\n"
                        
                        if len(docs) > 10:
                            result += f"\n... and {len(docs) - 10} more documents\n"
                        
                        result += "\n💡 Go to 'Chat' tab to query your knowledge base!"
                        
                        return result
                    
                    except Exception as e:
                        return f"❌ Error checking knowledge base: {str(e)}"
                
                verify_btn.click(verify_knowledge_base, outputs=verify_output)
                
                gr.Markdown("""
                ---
                ### 💡 What to Check:
                - Total number of documents
                - Document previews
                - Document IDs (for deletion)
                
                ### 🎯 Next Steps:
                1. Verify your documents are added
                2. Go to "Chat" tab
                3. Ask questions about your documents
                4. Use keywords like "my", "I am" for personal info
                """)
    
    # ==================== TAB 4: MANAGE KNOWLEDGE ====================
    with gr.Tab("🗑️ Manage"):
        gr.Markdown("# 🗑️ Manage Knowledge Base")
        gr.Markdown("**View, browse, and delete documents from your vector database**")
        
        with gr.Row():
            refresh_docs_btn = gr.Button("🔄 Refresh List", variant="secondary", size="sm")
        
        # Document list
        gr.Markdown("## 📚 Documents in Knowledge Base")
        
        doc_dataframe = gr.Dataframe(
            label="Documents",
            headers=["ID", "Preview"],
            interactive=False,
            wrap=True,
        )
        
        doc_info = gr.Textbox(
            label="Info",
            lines=2,
            interactive=False
        )
        
        # Load initial document list
        app.load(list_documents, outputs=[doc_dataframe, doc_info])
        
        gr.Markdown("---")
        gr.Markdown("## 🗑️ Delete Options")
        
        with gr.Tabs():
            with gr.Tab("Delete Specific"):
                gr.Markdown("**Delete a single document by ID**")
                
                delete_id_input = gr.Textbox(
                    label="Document ID",
                    placeholder="Paste the full document ID here...",
                    lines=1
                )
                
                delete_btn = gr.Button("🗑️ Delete Document", variant="stop", size="lg")
                delete_status = gr.Textbox(label="Status", lines=2)
                
                delete_btn.click(
                    delete_specific_document,
                    inputs=delete_id_input,
                    outputs=[delete_status, doc_dataframe, doc_info]
                )
            
            with gr.Tab("Clear All"):
                gr.Markdown("**⚠️ WARNING: This will delete ALL documents from your knowledge base!**")
                gr.Markdown("This action cannot be undone.")
                
                confirm_text = gr.Textbox(
                    label="Type 'DELETE ALL' to confirm",
                    placeholder="DELETE ALL",
                    lines=1
                )
                
                clear_all_btn = gr.Button("🗑️ Clear Entire Knowledge Base", variant="stop", size="lg")
                clear_status = gr.Textbox(label="Status", lines=2)
                
                def confirm_and_clear(confirmation):
                    if confirmation.strip() == "DELETE ALL":
                        return clear_all_knowledge()
                    else:
                        df, info = list_documents()
                        return "❌ Confirmation text does not match. Type exactly: DELETE ALL", df, info
                
                clear_all_btn.click(
                    confirm_and_clear,
                    inputs=confirm_text,
                    outputs=[clear_status, doc_dataframe, doc_info]
                )
        
        # Refresh button
        refresh_docs_btn.click(
            list_documents,
            outputs=[doc_dataframe, doc_info]
        )
        
        gr.Markdown("---")
        gr.Markdown("""
        ### 💡 Tips:
        - **View**: See all documents and their IDs
        - **Delete Specific**: Copy the full ID from the document list
        - **Clear All**: Use when you want to start fresh
        - **Refresh**: Update the list after adding/deleting documents
        """)
    
    # ==================== TAB 5: SHOPPING GUIDE ====================
    with gr.Tab("🛍️ Shopping"):
        gr.Markdown("# 🛍️ Smart Shopping Guide")
        
        gr.Markdown("""
        ## 📝 Command Examples:
        
        ### 🛒 E-Commerce:
        - `laptop on amazon`
        - `white shoes on ajio`
        - `red dress on myntra`
        - `makeup on nykaa`
        
        ### 🌐 Open Sites:
        - `open amazon`
        - `open flipkart`
        
        ### 🍕 Food:
        - `pizza on swiggy`
        - `biryani on zomato`
        
        ## ✅ Supported Platforms:
        **Shopping:** Amazon, Flipkart, Ajio, Myntra, Meesho, Nykaa, Snapdeal, TataCliq, JioMart
        
        **Food:** Swiggy, Zomato
        
        **Entertainment:** Netflix, Hotstar, Prime Video, BookMyShow
        """)
    
    # ==================== TAB 6: GUIDE ====================
    with gr.Tab("📖 Guide"):
        gr.Markdown("""
        # 📖 Complete User Guide
        
        ## 🎭 Real-Time Emotion Detection
        
        ### Live Camera:
        - **Start/Stop Camera** for live emotion detection
        - **7 Emotions:** Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
        - **Auto-refresh** for continuous monitoring
        - **Analytics** track last 50 detections
        - **TensorFlow CNN** model with temporal smoothing
        
        ### Static Image:
        - **Upload any image** with a visible face
        - **Instant detection** with annotated results
        - **Confidence scores** for each emotion
        - Works with **JPG, PNG** formats
        
        ### ✨ NEW: Empathetic Chat
        - If a strong emotion (Sad, Angry, etc.) is detected, the bot will ask you about it in the **Chat** tab.
        
        ## 💬 Chat Features
        
        ### Commands:
        - `weather in Mumbai` - Get weather report
        - `latest news` - Top headlines
        - `tell me a joke` - Random dad joke
        - `give me advice` - Life advice
        - `trending movies` - Current popular movies
        - `my ip address` - Find your IP
        
        ### Shopping:
        - `[product] on [website]` - Search products
        - `open [website]` - Open shopping sites
        
        ### Actions:
        - `open calculator` - Launch calculator
        - `open notepad` - Launch Notepad++
        - `send whatsapp` - Send WhatsApp message
        - `send email` - Send email
        
        ### Search:
        - `wikipedia [topic]` - Search Wikipedia
        - `play [video] on youtube` - Play YouTube video
        - `search [query] on google` - Google search
        
        ## 🎙️ Voice Features
        
        ### TTS (Text-to-Speech):
        Type text → Get voice response
        
        ### STT (Speech-to-Text):
        Record voice → Get text response
        
        ### STS (Speech-to-Speech):
        Record voice → Get voice response
        
        ## 📚 Knowledge Base Management
        
        ### Add Information:
        - **Text Notes:** Quick facts, personal info
        - **PDF Upload:** Documents, papers, books
        
        ### Manage Documents:
        - **View All:** See complete document list
        - **Delete Specific:** Remove individual documents
        - **Clear All:** Reset entire knowledge base
        - **Refresh:** Update document list
        
        ### Query Your Data:
        - Use keywords like "my", "I am", "my job"
        - Bot will search your personal knowledge base
        
        ## 🔧 Technologies
        
        - **LLM:** Groq (Llama 3.1 70B)
        - **Embeddings:** Sentence Transformers
        - **Emotion:** TensorFlow + OpenCV
        - **Search:** Tavily AI
        - **Voice:** Groq Whisper + gTTS
        - **UI:** Gradio 4.x
        - **Vector DB:** LlamaIndex
        
        ## 📦 Installation
        
        ```bash
        pip install gradio tensorflow opencv-python pandas textblob
        pip install llama-index llama-index-llms-groq
        pip install llama-index-embeddings-huggingface
        pip install groq tavily-python gtts pywhatkit
        pip install wikipedia-api python-decouple pillow
        ```
        
        ## ⚙️ .env Setup
        
        ```env
        GROQ_API_KEY=your_groq_key
        TAVILY_API_KEY=your_tavily_key
        NEWS_API_KEY=your_news_key
        OPENWEATHER_APP_ID=your_weather_key
        TMDB_API_KEY=your_tmdb_key
        EMAIL=your_email@gmail.com
        PASSWORD=your_app_password
        USER=YourName
        BOTNAME=Friday
        ```
        
        ## 💡 Tips
        
        1. **Emotion Detection:** 
           - Live camera works best in good lighting
           - Static images should have clear, front-facing faces
        
        2. **Knowledge Base:** 
           - Add personal info for better responses
           - Regularly manage documents to keep DB clean
        
        3. **Voice:** 
           - Speak clearly near microphone
           - Use quiet environment for best results
        
        4. **Shopping:** 
           - Supports 20+ e-commerce platforms
           - Works with food delivery and entertainment sites
        
        5. **Context:** 
           - Bot remembers last 10 conversations
           - Use "Reset Context" to start fresh
        
        ## 🆘 Troubleshooting
        
        - **No camera:** Check browser permissions
        - **Voice not working:** Allow microphone access
        - **Slow responses:** Check internet connection
        - **Emotion not detecting:** Ensure face is visible and well-lit
        - **Graph not showing:** Click "Update Analytics" after capturing emotions
        - **Delete not working:** Make sure to copy the complete document ID
        - **Static image fails:** Ensure image has a clear, visible face
        - **Knowledge base empty:** Check console logs for errors during document addition
        - **Query returns empty:** Try asking differently or add more specific information
        
        ---
        
        **Made with ❤️ | Powered by AI | Full-Featured Assistant Suite**
        """)
    
    # ==================== CROSS-TAB EVENT LISTENERS ====================
    chat_tab.select(
        fn=check_emotion_trigger_and_update_chat,
        inputs=chatbot_display,
        outputs=chatbot_display
    )

# ==============================================================================
# ------------------ LAUNCH APPLICATION ----------------------------------------
# ==============================================================================

if __name__ == "__main__":
    try:
        print("\n🌐 Launching Gradio Interface...")
        print("="*70)
        
        app.launch(
            share=False,
            inbrowser=True,
            show_error=True,
            server_port=7860
        )
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 Application stopped by user (Ctrl+C)")
        if emotion_detector:
            emotion_detector.stop_camera()
        print("="*70)
    
    except Exception as e:
        print("\n\n" + "="*70)
        print(f"❌ Error launching application:")
        print("="*70)
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Check port 7860 is available")
        print("   2. Verify all dependencies installed")
        print("   3. Check .env file exists with API keys")
        print("="*70)
