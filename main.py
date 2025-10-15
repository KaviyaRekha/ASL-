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

# Continue with the rest of your existing main.py code...
# [PASTE ALL YOUR EXISTING CODE AFTER THIS LINE]
