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

# ==================== DATABASE CONFIGURATION ====================
import asyncpg
from datetime import datetime, timedelta
import json
import uuid

class DatabaseManager:
    def __init__(self):
        self.pool = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            print(f"🔗 Connecting to database at {DATABASE_CONFIG['host']}...")
            
            # PostgreSQL connection - USING THE DOCKER CONFIG WE ADDED AT TOP
            self.pool = await asyncpg.create_pool(
                user=DATABASE_CONFIG['user'],
                password=DATABASE_CONFIG['password'],
                database=DATABASE_CONFIG['database'],
                host=DATABASE_CONFIG['host'],
                port=DATABASE_CONFIG['port']
            )
            
            # Create tables if they don't exist
            await self.create_tables()
            print("✅ Database connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("⚠️  Running in offline mode without database")
            return False
