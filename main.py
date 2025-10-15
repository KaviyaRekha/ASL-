import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyttsx3
import time
import webbrowser
import math
import speech_recognition as sr
import os
import pickle
import json
import subprocess
import pygetwindow as gw
import requests
import smtplib
from email.mime.text import MimeText
import threading

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

# ==================== CONFIGURATION ====================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# ... rest of your existing code continues ...
