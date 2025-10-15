import os
import sys

# ==================== DOCKER CONFIGURATION ====================
print("🔍 Checking environment...")

# Docker-specific configurations
if os.path.exists('/.dockerenv'):
    print("🐳 Running in Docker container")
    # Docker-specific settings
    os.environ['DISPLAY'] = ':0'
    
    # Database configuration for Docker
    DATABASE_CONFIG = {
        'user': 'signlink_user',
        'password': 'secure_password',
        'database': 'signlink_db',
        'host': 'postgres',  # Use service name from docker-compose
        'port': 5432
    }
else:
    print("💻 Running in local environment")
    # Local development configuration
    DATABASE_CONFIG = {
        'user': 'signlink_user',
        'password': 'secure_password', 
        'database': 'signlink_db',
        'host': 'localhost',
        'port': 5432
    }

print(f"📊 Database host: {DATABASE_CONFIG['host']}")
# ==================== END DOCKER CONFIG ====================

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyttsx3
import time
import webbrowser
import math
import speech_recognition as sr
import pickle
import json
import subprocess
import pygetwindow as gw
import requests
import smtplib
from email.mime.text import MimeText
import threading
import asyncpg
from datetime import datetime, timedelta
import uuid

# ==================== CONFIGURATION ====================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TYPING_CONFIRM_TIME = 1.5
MODE_SWITCH_TIME = 2.5
CLICK_THRESHOLD = 40
FEEDBACK_DURATION = 2
MOUSE_MOVE_THRESHOLD = 50

# ==================== ENTERPRISE SECTOR CONFIGURATION ====================
DEMO_MODE = True
CURRENT_SCENARIO = "Enterprise - Manufacturing Control & Productivity"
ACCURACY_DISPLAY = "97%"
LATENCY_DISPLAY = "65ms"
DEMO_SECTOR = "enterprise"  
INPUT_MODE = "quick_access"

# ==================== CUSTOM NUMPY SCALER ====================
class NumpyScaler:
    def __init__(self, mean=None, scale=None):
        self.mean = mean
        self.scale = scale

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale if self.mean is not None else X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ==================== CUSTOM NUMPY KNN CLASSIFIER ====================
class NumpyKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        dists = np.linalg.norm(self.X_train - X, axis=1)
        k_idx = np.argsort(dists)[:self.k]
        k_labels = self.y_train[k_idx]
        values, counts = np.unique(k_labels, return_counts=True)
        return [values[np.argmax(counts)]]

# ==================== MODEL LOADING SECTION ====================
def load_pickle(file, default=None):
    path = os.path.join(MODEL_DIR, file)
    print(f"📂 Loading model: {path}")
    
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            print(f"✅ Successfully loaded {file}")
            return model
        except Exception as e:
            print(f"❌ Could not load {file}: {e}")
            return default
    else:
        print(f"⚠️  Model file not found: {path}")
        print(f"💡 Run: python models/create_placeholder_models.py")
        return default

# Load models with error handling
print("🔄 Loading machine learning models...")
knn_update = load_pickle("models/asl_update_knn.pkl")
knn_base = load_pickle("models/asl_knn.pkl") 
scaler = load_pickle("models/asl_scaler.pkl", default=NumpyScaler())
label_encoder = load_pickle("models/asl_label_encoder.pkl")

# Check if models loaded successfully
if knn_base is None or label_encoder is None:
    print("🚨 Critical models missing! Using fallback mode.")
    print("💡 Please run: python models/create_placeholder_models.py")
    # Create fallback label encoder for basic functionality
    class FallbackLabelEncoder:
        def __init__(self):
            self.classes_ = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        
        def inverse_transform(self, labels):
            if isinstance(labels, (list, np.ndarray)):
                return [self.classes_[label] if label < len(self.classes_) else 'A' for label in labels]
            else:
                return self.classes_[labels] if labels < len(self.classes_) else 'A'
    
    label_encoder = FallbackLabelEncoder()
    print("🔄 Using fallback label encoder for basic functionality")

print("✅ Model loading completed!")

# ==================== CONTINUE WITH YOUR EXISTING CODE ====================

# SMS/Email Configuration (Replace with your actual credentials)
SMS_CONFIG = {
    'twilio_account_sid': 'your_account_sid',
    'twilio_auth_token': 'your_auth_token',
    'twilio_number': '+1234567890',
    'admin_mobile': '+0987654321'
}

EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'your_email@gmail.com',
    'password': 'your_password',
    'admin_email': 'admin@hospital.com'
}

# SECTOR-SPECIFIC COLOR SCHEMES
COLORS = {
    "healthcare": {
        "primary": (0, 255, 255),    # Bright Cyan - Medical/clean
        "secondary": (255, 255, 0),  # Bright Yellow - Attention/caution
        "accent": (0, 255, 0),       # Bright Green - Safe/go
        "text": (255, 255, 255),     # White - Clear visibility
        "highlight": (255, 165, 0),  # Orange - Important alerts
        "button_bg": (0, 100, 255),  # Blue - Trust/calm
        "key_bg": (30, 30, 100),     
        "key_border": (0, 255, 255),
        "accessibility": (255, 105, 180)  # Hot Pink - Accessibility focus
    },
    "enterprise": {
        "primary": (255, 100, 255),  # Bright Pink - Professional/Modern
        "secondary": (100, 255, 255),# Bright Light Blue - Technology
        "accent": (255, 255, 100),   # Bright Light Yellow - Attention
        "text": (255, 255, 255),     # White - Clear text
        "highlight": (255, 165, 0),  # Orange - Important actions
        "button_bg": (255, 100, 100),# Bright Red - Critical functions
        "key_bg": (100, 30, 100),    # Dark Purple - Professional
        "key_border": (255, 100, 255), # Pink borders
        "accessibility": (100, 255, 100)  # Light Green - Safety/Go
    },
    "education": {
        "primary": (255, 165, 0),    # Bright Orange - Energetic/engaging
        "secondary": (144, 238, 144),# Light Green - Growth/learning
        "accent": (255, 255, 100),   # Bright Light Yellow - Highlight/important
        "text": (255, 255, 255),     # White - Clear readability
        "highlight": (0, 255, 255),  # Cyan - Interactive elements
        "button_bg": (100, 255, 100),# Bright Green - Positive actions
        "key_bg": (30, 100, 30),     # Dark Green - Educational
        "key_border": (255, 165, 0), # Orange borders
        "accessibility": (255, 105, 180)  # Pink - Special needs focus
    }
}

# SECTOR-SPECIFIC QUICK ACCESS BUTTONS
quick_buttons = {
    "healthcare": [
        {'name': 'PATIENT INFO', 'action': 'patient_info', 'gesture': 'P', 'color': (0, 255, 255)},
        {'name': 'MED CHART', 'action': 'medical_chart', 'gesture': 'M', 'color': (255, 255, 0)},
        {'name': 'EMERGENCY', 'action': 'emergency_call', 'gesture': 'E', 'color': (255, 0, 0)},
        {'name': 'COMMUNICATE', 'action': 'healthcare_communicate', 'gesture': 'C', 'color': (0, 255, 0)},
        {'name': 'VOICE CMD', 'action': 'voice_commands', 'gesture': 'V', 'color': (255, 105, 180)}
    ],
    "enterprise": [
        {'name': 'DASHBOARD', 'action': 'control_dashboard', 'gesture': 'D', 'color': (255, 100, 255)},
        {'name': 'CAD', 'action': 'cad_control', 'gesture': 'C', 'color': (100, 255, 255)},
        {'name': 'PRESENT', 'action': 'presentation_mode', 'gesture': 'P', 'color': (255, 255, 100)},
        {'name': 'MONITORS', 'action': 'multi_monitor', 'gesture': 'M', 'color': (255, 100, 100)},
        {'name': 'VOICE CMD', 'action': 'voice_commands', 'gesture': 'V', 'color': (100, 255, 100)}
    ],
    "education": [
        {'name': 'LESSON', 'action': 'lesson_control', 'gesture': 'L', 'color': (255, 165, 0)},
        {'name': 'WHITEBOARD', 'action': 'digital_whiteboard', 'gesture': 'W', 'color': (144, 238, 144)},
        {'name': 'ASSESSMENT', 'action': 'assessment_tool', 'gesture': 'A', 'color': (255, 255, 100)},
        {'name': 'ACCESSIBILITY', 'action': 'accessibility_tools', 'gesture': 'X', 'color': (0, 255, 255)},
        {'name': 'VOICE CMD', 'action': 'voice_commands', 'gesture': 'V', 'color': (255, 105, 180)}
    ]
}

# [CONTINUE WITH THE REST OF YOUR EXISTING CODE...]
# Paste all your remaining classes and functions here:
# - DatabaseManager class
# - EnhancedSignLinkSystem class  
# - GestureSequenceManager class
# - SoftwareHapticEmotionEngine class
# - Advanced3DSpatialModel class
# - PersonalizedGestureModel class
# - IntelligentChatbot class
# - SectorSpecificUI class
# - All your action functions
# - Main loop function
