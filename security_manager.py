import cv2
import numpy as np
import threading
import time
import os
import logging
import json
import smtplib
import requests
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, List, Optional, Any
import sqlite3
import hashlib
import base64

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import SECURITY_CONFIG, ALERT_CONFIG, DISPLAY_CONFIG, RECORDING_CONFIG, get_wib_timestamp, get_wib_filename_timestamp

logger = logging.getLogger(__name__)

class WebhookHandler:
    """Handle various webhook services for security alerts"""

    def __init__(self):
        self.webhook_config = ALERT_CONFIG.get('webhook_alerts', {})
        self.enabled = self.webhook_config.get('enabled', False)
        self.retry_attempts = self.webhook_config.get('retry_attempts', 3)
        self.timeout = self.webhook_config.get('timeout', 10)
        self.last_alert_time = {}
        self.cooldown = self.webhook_config.get('alert_cooldown', 30)

        logger.info(f"🔗 Webhook Handler initialized - Enabled: {self.enabled}")

    def send_alert(self, camera_name: str, detection_type: str, confidence: float,
                   image_path: str = None, priority: str = "medium") -> bool:
        """Send alert to all enabled webhook services"""
        if not self.enabled:
            return False

        # Check cooldown
        alert_key = f"{camera_name}_{detection_type}"
        current_time = time.time()

        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.cooldown:
                logger.debug(f"Webhook alert on cooldown for {alert_key}")
                return False

        self.last_alert_time[alert_key] = current_time

        # Send to all enabled services
        services = self.webhook_config.get('services', {})
        success = False

        for service_name, service_config in services.items():
            if service_config.get('enabled', False):
                try:
                    if service_name == 'discord':
                        success |= self._send_discord(camera_name, detection_type, confidence, image_path, priority)
                    elif service_name == 'slack':
                        success |= self._send_slack(camera_name, detection_type, confidence, image_path, priority)
                    elif service_name == 'teams':
                        success |= self._send_teams(camera_name, detection_type, confidence, image_path, priority)
                    elif service_name == 'custom':
                        success |= self._send_custom(camera_name, detection_type, confidence, image_path, priority)

                except Exception as e:
                    logger.error(f"❌ Webhook error for {service_name}: {e}")

        return success

    def _send_discord(self, camera_name: str, detection_type: str, confidence: float,
                     image_path: str = None, priority: str = "medium") -> bool:
        """Send alert to Discord webhook"""
        config = self.webhook_config['services']['discord']
        webhook_url = config.get('webhook_url')

        if not webhook_url or webhook_url == 'https://discord.com/api/webhooks/YOUR_WEBHOOK_URL':
            logger.warning("Discord webhook URL not configured")
            return False

        # Priority colors
        colors = {
            'critical': 0xff0000,  # Red
            'high': 0xff8800,      # Orange
            'medium': 0xffff00,    # Yellow
            'low': 0x00ff00        # Green
        }

        timestamp = get_wib_timestamp()

        # Create Discord embed
        embed = {
            "title": "🚨 Security Alert",
            "description": f"**{detection_type.upper()}** detected at **{camera_name}**",
            "color": colors.get(priority, config.get('color', 0xff0000)),
            "fields": [
                {
                    "name": "📍 Location",
                    "value": camera_name,
                    "inline": True
                },
                {
                    "name": "🎯 Detection",
                    "value": f"{detection_type} ({confidence:.1%})",
                    "inline": True
                },
                {
                    "name": "⚠️ Priority",
                    "value": priority.upper(),
                    "inline": True
                },
                {
                    "name": "🕒 Time (WIB)",
                    "value": timestamp,
                    "inline": False
                }
            ],
            "footer": {
                "text": "Professional Security System",
                "icon_url": config.get('avatar_url', '')
            },
            "timestamp": get_wib_timestamp("%Y-%m-%dT%H:%M:%S+07:00")
        }

        # Prepare payload
        payload = {
            "username": config.get('username', 'Security System'),
            "avatar_url": config.get('avatar_url', ''),
            "embeds": [embed]
        }

        # Add mention if configured
        mention_role = config.get('mention_role_id')
        if mention_role:
            payload["content"] = f"<@&{mention_role}>"

        try:
            # Send webhook without image first
            response = requests.post(webhook_url, json=payload, timeout=self.timeout)

            if response.status_code in [200, 204]:
                logger.info(f"✅ Discord alert sent for {camera_name}")

                # Send image as follow-up if available and enabled
                if image_path and config.get('send_image', True) and os.path.exists(image_path):
                    self._send_discord_image(webhook_url, config, image_path, camera_name)

                return True
            else:
                logger.error(f"❌ Discord webhook failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Discord webhook error: {e}")
            return False

    def _send_discord_image(self, webhook_url: str, config: dict, image_path: str, camera_name: str):
        """Send image to Discord as follow-up message"""
        try:
            with open(image_path, 'rb') as f:
                files = {
                    'file': (f'alert_{camera_name}_{int(time.time())}.jpg', f, 'image/jpeg')
                }

                payload = {
                    "username": config.get('username', 'Security System')
                    # "content": f"📸 Alert screenshot from {camera_name}"
                }

                response = requests.post(webhook_url, data=payload, files=files, timeout=self.timeout)

                if response.status_code in [200, 204]:
                    logger.info(f"✅ Discord image sent for {camera_name}")
                else:
                    logger.warning(f"⚠️ Discord image failed: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Discord image error: {e}")

    def _send_slack(self, camera_name: str, detection_type: str, confidence: float,
                   image_path: str = None, priority: str = "medium") -> bool:
        """Send alert to Slack webhook"""
        config = self.webhook_config['services']['slack']
        webhook_url = config.get('webhook_url')

        if not webhook_url or 'YOUR' in webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        timestamp = get_wib_timestamp()

        # Priority colors
        colors = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'warning',
            'low': 'good'
        }

        payload = {
            "channel": config.get('channel', '#security-alerts'),
            "username": config.get('username', 'Security Bot'),
            "icon_emoji": config.get('icon_emoji', ':warning:'),
            "attachments": [
                {
                    "color": colors.get(priority, 'warning'),
                    "title": "🚨 Security Alert",
                    "fields": [
                        {
                            "title": "Location",
                            "value": camera_name,
                            "short": True
                        },
                        {
                            "title": "Detection",
                            "value": f"{detection_type} ({confidence:.1%})",
                            "short": True
                        },
                        {
                            "title": "Priority",
                            "value": priority.upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": timestamp,
                            "short": True
                        }
                    ],
                    "footer": "Professional Security System",
                    "ts": int(time.time())
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                logger.info(f"✅ Slack alert sent for {camera_name}")
                return True
            else:
                logger.error(f"❌ Slack webhook failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Slack webhook error: {e}")
            return False

    def _send_teams(self, camera_name: str, detection_type: str, confidence: float,
                   image_path: str = None, priority: str = "medium") -> bool:
        """Send alert to Microsoft Teams webhook"""
        config = self.webhook_config['services']['teams']
        webhook_url = config.get('webhook_url')

        if not webhook_url or 'YOUR' in webhook_url:
            logger.warning("Teams webhook URL not configured")
            return False

        timestamp = get_wib_timestamp()

        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": config.get('theme_color', 'FF0000'),
            "summary": "Security Alert",
            "sections": [
                {
                    "activityTitle": config.get('title', 'Security Alert'),
                    "activitySubtitle": f"{detection_type.upper()} detected at {camera_name}",
                    "facts": [
                        {
                            "name": "Location:",
                            "value": camera_name
                        },
                        {
                            "name": "Detection:",
                            "value": f"{detection_type} ({confidence:.1%})"
                        },
                        {
                            "name": "Priority:",
                            "value": priority.upper()
                        },
                        {
                            "name": "Time:",
                            "value": timestamp
                        }
                    ],
                    "markdown": True
                }
            ]
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                logger.info(f"✅ Teams alert sent for {camera_name}")
                return True
            else:
                logger.error(f"❌ Teams webhook failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Teams webhook error: {e}")
            return False

    def _send_custom(self, camera_name: str, detection_type: str, confidence: float,
                    image_path: str = None, priority: str = "medium") -> bool:
        """Send alert to custom webhook"""
        config = self.webhook_config['services']['custom']
        webhook_url = config.get('webhook_url')

        if not webhook_url or 'your-custom' in webhook_url:
            logger.warning("Custom webhook URL not configured")
            return False

        timestamp = get_wib_timestamp()

        # Custom payload format
        payload = {
            "event": "security_alert",
            "timestamp": timestamp,
            "camera": camera_name,
            "detection": {
                "type": detection_type,
                "confidence": confidence,
                "priority": priority
            },
            "system": "Professional Security System"
        }

        # Add base64 encoded image if available
        if image_path and config.get('send_image', True) and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    payload["image"] = {
                        "format": "jpeg",
                        "data": image_data
                    }
            except Exception as e:
                logger.warning(f"⚠️ Custom webhook image encoding failed: {e}")

        try:
            headers = config.get('headers', {'Content-Type': 'application/json'})
            method = config.get('method', 'POST').upper()

            if method == 'POST':
                response = requests.post(webhook_url, json=payload, headers=headers, timeout=self.timeout)
            elif method == 'PUT':
                response = requests.put(webhook_url, json=payload, headers=headers, timeout=self.timeout)
            else:
                logger.error(f"❌ Unsupported HTTP method: {method}")
                return False

            if response.status_code in [200, 201, 202, 204]:
                logger.info(f"✅ Custom webhook alert sent for {camera_name}")
                return True
            else:
                logger.error(f"❌ Custom webhook failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Custom webhook error: {e}")
            return False

class SecurityManager:
    def __init__(self):
        self.armed = SECURITY_CONFIG['armed']
        self.sensitivity = SECURITY_CONFIG['sensitivity']
        self.alert_cooldown = SECURITY_CONFIG['alert_cooldown']
        self.recording_duration = SECURITY_CONFIG['recording_duration']

        # Alert tracking
        self.last_alerts = {}  # camera_id -> timestamp
        self.active_recordings = {}  # camera_id -> VideoWriter
        self.alert_history = []

        # Database for security logs
        self.db_path = os.path.join(DISPLAY_CONFIG['logs_dir'], 'security.db')
        self.init_database()

        # Create directories
        for directory in ['logs_dir', 'recordings_dir', 'alerts_dir']:
            os.makedirs(DISPLAY_CONFIG[directory], exist_ok=True)

        # Sound alert
        self.sound_enabled = ALERT_CONFIG['sound_alerts']['enabled']

        # Initialize components
        self.webhook_handler = WebhookHandler()

        # Recording auto-cleanup (10 minutes)
        self.recording_files = {}  # {file_path: creation_time}
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._recording_cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info("Security Manager initialized")
        logger.info("🗑️ Auto-cleanup: Recordings akan dihapus setelah 10 menit")

    def _recording_cleanup_loop(self):
        """Background loop untuk auto-delete recordings setelah 10 menit"""
        while self.cleanup_running:
            try:
                current_time = time.time()
                cleanup_threshold = 10 * 60  # 10 minutes in seconds

                # Check tracked recording files
                files_to_remove = []
                for file_path, creation_time in self.recording_files.items():
                    if current_time - creation_time > cleanup_threshold:
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                logger.info(f"🗑️ Auto-deleted recording: {os.path.basename(file_path)}")
                            except Exception as e:
                                logger.error(f"❌ Failed to delete {file_path}: {e}")
                        files_to_remove.append(file_path)

                # Remove from tracking
                for file_path in files_to_remove:
                    del self.recording_files[file_path]

                # Also scan recordings directory for any untracked old files
                try:
                    recordings_dir = DISPLAY_CONFIG['recordings_dir']
                    for filename in os.listdir(recordings_dir):
                        if filename.startswith('REC_') and filename.endswith('.mp4'):
                            file_path = os.path.join(recordings_dir, filename)
                            file_age = current_time - os.path.getctime(file_path)

                            if file_age > cleanup_threshold:
                                try:
                                    os.remove(file_path)
                                    logger.info(f"🗑️ Auto-deleted old recording: {filename}")
                                except Exception as e:
                                    logger.error(f"❌ Failed to delete old recording {filename}: {e}")
                except Exception as e:
                    logger.error(f"❌ Error scanning recordings directory: {e}")

            except Exception as e:
                logger.error(f"❌ Recording cleanup error: {e}")

            # Check every minute
            time.sleep(60)

    def init_database(self):
        """Initialize security database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detected_objects TEXT,
                    confidence REAL,
                    priority TEXT,
                    zone TEXT,
                    alert_sent BOOLEAN,
                    screenshot_path TEXT,
                    recording_path TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    camera_id TEXT
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")

    def log_event(self, camera_id: str, camera_name: str, event_type: str,
                  detected_objects: List = None, confidence: float = 0.0,
                  priority: str = "medium", zone: str = "unknown",
                  alert_sent: bool = False, screenshot_path: str = None,
                  recording_path: str = None):
        """Log security event to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = get_wib_timestamp("%Y-%m-%dT%H:%M:%S")
            objects_json = json.dumps(detected_objects) if detected_objects else None

            cursor.execute('''
                INSERT INTO security_events
                (timestamp, camera_id, camera_name, event_type, detected_objects,
                 confidence, priority, zone, alert_sent, screenshot_path, recording_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, camera_id, camera_name, event_type, objects_json,
                  confidence, priority, zone, alert_sent, screenshot_path, recording_path))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Event logging error: {str(e)}")

    def process_detection(self, camera_id: str, camera_name: str, detections: List,
                         frame: np.ndarray, camera_config: Dict) -> bool:
        """Process detection and trigger alerts if needed"""
        if not self.armed or not detections:
            return False

        current_time = time.time()
        zone = camera_config.get('security_zone', 'unknown')
        priority = camera_config.get('priority', 'medium')

        # Check cooldown
        if camera_id in self.last_alerts:
            if current_time - self.last_alerts[camera_id] < self.alert_cooldown:
                return False

        # Filter critical detections
        critical_detections = []
        for detection in detections:
            if detection['class'] in ['person', 'car', 'motorcycle']:
                critical_detections.append(detection)

        if not critical_detections:
            return False

        # Update alert time
        self.last_alerts[camera_id] = current_time

        # Generate alert
        alert_triggered = self.trigger_alert(
            camera_id, camera_name, critical_detections, frame,
            zone, priority
        )

        # Log event
        self.log_event(
            camera_id=camera_id,
            camera_name=camera_name,
            event_type="detection_alert",
            detected_objects=[d['class'] for d in critical_detections],
            confidence=max(d['confidence'] for d in critical_detections),
            priority=priority,
            zone=zone,
            alert_sent=alert_triggered
        )

        return alert_triggered

    def trigger_alert(self, camera_id: str, camera_name: str, detections: List,
                     frame: np.ndarray, zone: str, priority: str) -> bool:
        """Trigger comprehensive alert system with WIB timezone"""
        wib_timestamp = get_wib_timestamp("%Y-%m-%dT%H:%M:%S")
        alert_id = hashlib.md5(f"{camera_id}_{wib_timestamp}".encode()).hexdigest()[:8]

        # 1. Save alert screenshot FIRST (16:9 optimized)
        screenshot_path = self.save_alert_screenshot(frame, camera_name, detections, alert_id)

        # 2. Start recording if enabled
        recording_path = None
        if RECORDING_CONFIG['enabled']:
            recording_path = self.start_alert_recording(camera_id, camera_name, alert_id)

        # 3. Sound alert
        if self.sound_enabled:
            self.play_alert_sound()

        # 4. Send alerts in order: Image first, then video (if available)
        # Webhook alerts (Discord, Slack, Teams, Custom) - Send image immediately
        if self.webhook_handler.enabled:
            self.webhook_handler.send_alert(
                camera_name, detections[0]['class'], max(d['confidence'] for d in detections),
                screenshot_path, priority
            )

        # 5. Email alert
        if ALERT_CONFIG['email_alerts']['enabled']:
            self.send_email_alert(camera_name, detections, screenshot_path, zone, priority)

        # 6. Telegram alert with priority
        if ALERT_CONFIG['telegram_bot']['enabled']:
            self.send_telegram_alert(camera_name, detections, screenshot_path, zone, priority)

        # Add to alert history with recording info
        alert_info = {
            'id': alert_id,
            'timestamp': wib_timestamp,
            'camera_id': camera_id,
            'camera_name': camera_name,
            'detections': [d['class'] for d in detections],
            'zone': zone,
            'priority': priority,
            'screenshot': screenshot_path,
            'recording': recording_path,
            'recording_pending': recording_path is not None
        }
        self.alert_history.append(alert_info)

        # Keep only last 100 alerts in memory
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        logger.warning(f"🚨 SECURITY ALERT [{alert_id}]: {camera_name} - {[d['class'] for d in detections]} (WIB: {get_wib_timestamp()})")

        return True

    def save_alert_screenshot(self, frame: np.ndarray, camera_name: str,
                            detections: List, alert_id: str) -> str:
        """Save alert screenshot in perfect 16:9 aspect ratio"""
        try:
            # Ensure 16:9 aspect ratio
            target_height = 1080
            target_width = 1920

            # Resize frame to 16:9
            resized_frame = cv2.resize(frame, (target_width, target_height))

            # Add professional alert overlay
            overlay_height = 120
            overlay = np.zeros((overlay_height, target_width, 3), dtype=np.uint8)
            overlay[:] = (0, 0, 50)  # Dark red background

            # Alert header
            cv2.putText(overlay, "SECURITY ALERT", (50, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Camera and timestamp (WIB)
            timestamp = get_wib_timestamp()
            cv2.putText(overlay, f"{camera_name} | {timestamp} WIB", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Detections
            detection_text = " | ".join([d['class'].upper() for d in detections])
            cv2.putText(overlay, f"DETECTED: {detection_text}", (50, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Alert ID (top right)
            cv2.putText(overlay, f"ID: {alert_id}", (target_width - 200, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Combine overlay with frame
            final_frame = np.vstack([overlay, resized_frame[:-overlay_height]])

            # Save with high quality
            timestamp_str = get_wib_filename_timestamp()
            filename = f"{DISPLAY_CONFIG['alerts_dir']}/ALERT_{camera_name}_{alert_id}_{timestamp_str}.jpg"

            cv2.imwrite(filename, final_frame, [cv2.IMWRITE_JPEG_QUALITY, DISPLAY_CONFIG['compression_quality']])

            return filename

        except Exception as e:
            logger.error(f"Screenshot save error: {str(e)}")
            return None

    def start_alert_recording(self, camera_id: str, camera_name: str, alert_id: str) -> str:
        """Start recording after detection and return filename"""
        try:
            if camera_id in self.active_recordings:
                return self.active_recordings[camera_id]['filename']  # Already recording

            timestamp = get_wib_filename_timestamp()
            filename = f"{DISPLAY_CONFIG['recordings_dir']}/REC_{camera_name}_{alert_id}_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = RECORDING_CONFIG['fps']
            resolution = RECORDING_CONFIG['resolution']

            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

            if writer.isOpened():
                current_time = time.time()
                self.active_recordings[camera_id] = {
                    'writer': writer,
                    'start_time': current_time,
                    'filename': filename,
                    'alert_id': alert_id,
                    'camera_name': camera_name
                }

                # Track file for auto-cleanup (10 minutes)
                self.recording_files[filename] = current_time

                logger.info(f"📹 Started recording: {filename}")
                logger.info(f"🗑️ Will auto-delete in 10 minutes: {os.path.basename(filename)}")
                return filename
            else:
                logger.error(f"❌ Failed to open video writer for {filename}")
                return None

        except Exception as e:
            logger.error(f"❌ Recording start error: {str(e)}")
            return None

    def update_recording(self, camera_id: str, frame: np.ndarray):
        """Update active recording with new frame"""
        if camera_id not in self.active_recordings:
            return

        try:
            recording = self.active_recordings[camera_id]
            current_time = time.time()

            # Check if recording duration exceeded
            if current_time - recording['start_time'] > self.recording_duration:
                self.stop_recording(camera_id)
                return

            # Resize frame to recording resolution
            resolution = RECORDING_CONFIG['resolution']
            resized_frame = cv2.resize(frame, resolution)

            recording['writer'].write(resized_frame)

        except Exception as e:
            logger.error(f"Recording update error: {str(e)}")
            self.stop_recording(camera_id)

    def stop_recording(self, camera_id: str):
        """Stop active recording and send to webhook"""
        if camera_id not in self.active_recordings:
            return

        try:
            recording = self.active_recordings[camera_id]
            recording['writer'].release()

            logger.info(f"📹 Recording saved: {recording['filename']}")

            # Send recording to Discord webhook if enabled
            if self.webhook_handler.enabled and os.path.exists(recording['filename']):
                self.send_recording_to_webhook(
                    recording['filename'],
                    recording['camera_name'],
                    recording['alert_id']
                )

            # Send recording to Telegram if enabled
            if ALERT_CONFIG['telegram_bot']['enabled'] and os.path.exists(recording['filename']):
                self.send_telegram_video(
                    recording['filename'],
                    recording['camera_name'],
                    recording['alert_id']
                )

            # Update database with recording path
            self.log_event(
                camera_id=camera_id,
                camera_name=recording.get('camera_name', ''),
                event_type="recording_completed",
                recording_path=recording['filename']
            )

            del self.active_recordings[camera_id]

        except Exception as e:
            logger.error(f"❌ Recording stop error: {str(e)}")

    def send_recording_to_webhook(self, recording_path: str, camera_name: str, alert_id: str):
        """Send recording video to webhook (Discord)"""
        try:
            # Only send to Discord for now (has good video support)
            discord_config = self.webhook_handler.webhook_config.get('services', {}).get('discord', {})

            if not discord_config.get('enabled', False):
                return

            webhook_url = discord_config.get('webhook_url')
            if not webhook_url or 'YOUR' in webhook_url:
                return

            # Check file size (Discord limit is 25MB for webhooks)
            file_size = os.path.getsize(recording_path) / (1024 * 1024)  # MB
            if file_size > 24:  # Leave some margin
                logger.warning(f"⚠️ Recording too large for Discord: {file_size:.1f}MB")
                return

            with open(recording_path, 'rb') as f:
                files = {
                    'file': (f'recording_{camera_name}_{alert_id}.mp4', f, 'video/mp4')
                }

                payload = {
                    "username": discord_config.get('username', 'Security System'),
                    # "content": f"📹 **Security Recording** from **{camera_name}** (Alert ID: {alert_id})\n🕒 Time: {get_wib_timestamp()} WIB"
                }

                response = requests.post(webhook_url, data=payload, files=files, timeout=30)

                if response.status_code in [200, 204]:
                    logger.info(f"✅ Recording sent to Discord: {camera_name}")
                else:
                    logger.warning(f"⚠️ Discord recording failed: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Recording webhook error: {e}")

    def play_alert_sound(self):
        """Play alert sound"""
        try:
            if os.path.exists(ALERT_CONFIG['sound_alerts']['alert_sound']):
                # Async sound playing to not block
                threading.Thread(target=self._play_sound_async, daemon=True).start()
            else:
                # Fallback: system beep
                print('\a')  # System beep
        except Exception as e:
            logger.error(f"Sound alert error: {str(e)}")

    def _play_sound_async(self):
        """Play sound asynchronously"""
        try:
            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(ALERT_CONFIG['sound_alerts']['alert_sound'])
                pygame.mixer.music.set_volume(ALERT_CONFIG['sound_alerts']['volume'])
                pygame.mixer.music.play()
            else:
                # Fallback if pygame not available
                pass
        except Exception as e:
            logger.error(f"Async sound error: {str(e)}")

    def send_email_alert(self, camera_name: str, detections: List,
                        screenshot_path: str, zone: str, priority: str):
        """Send email alert with screenshot"""
        try:
            config = ALERT_CONFIG['email_alerts']

            msg = MIMEMultipart()
            msg['From'] = config['email']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"🚨 SECURITY ALERT - {camera_name} ({priority.upper()})"

            # Email body
            detected_objects = ', '.join([d['class'].title() for d in detections])
            timestamp = get_wib_timestamp()

            body = f"""
SECURITY ALERT DETECTED

📅 Time: {timestamp} WIB
📹 Camera: {camera_name}
🔍 Zone: {zone.title()}
⚠️ Priority: {priority.upper()}
🎯 Detected: {detected_objects}

This is an automated alert from your home security system.
Please check your cameras immediately.

---
Home Security System
            """

            msg.attach(MIMEText(body, 'plain'))

            # Attach screenshot if available
            if screenshot_path and os.path.exists(screenshot_path):
                with open(screenshot_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', f'attachment; filename="alert_{camera_name}.jpg"')
                    msg.attach(image)

            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['email'], config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent for {camera_name}")

        except Exception as e:
            logger.error(f"Email alert error: {str(e)}")

    def send_telegram_alert(self, camera_name: str, detections: List,
                           screenshot_path: str, zone: str, priority: str = "medium"):
        """Send optimized Telegram alert with photo and video support"""
        try:
            config = ALERT_CONFIG['telegram_bot']
            bot_token = config['bot_token']
            chat_id = config['chat_id']

            detected_objects = ', '.join([d['class'].title() for d in detections])
            timestamp = get_wib_timestamp()

            # Priority emojis
            priority_emojis = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }

            priority_emoji = priority_emojis.get(priority, '🟡')

            # Enhanced message with better formatting
            message = f"""🚨 *SECURITY ALERT* {priority_emoji}

📹 *Camera:* {camera_name}
🔍 *Zone:* {zone.title()}
🎯 *Detected:* {detected_objects}
⚠️ *Priority:* {priority.upper()}
📅 *Time:* {timestamp} WIB

🏠 _Professional Home Security System_"""

            # Send photo with enhanced caption (if enabled)
            if screenshot_path and os.path.exists(screenshot_path) and config.get('send_photo', True):
                success = self._send_telegram_photo(bot_token, chat_id, screenshot_path, message)

                if success:
                    logger.info(f"✅ Telegram photo alert sent for {camera_name}")

                    # Store alert info for potential video follow-up
                    alert_key = f"{camera_name}_{int(time.time())}"
                    if not hasattr(self, 'telegram_pending_videos'):
                        self.telegram_pending_videos = {}

                    self.telegram_pending_videos[alert_key] = {
                        'camera_name': camera_name,
                        'bot_token': bot_token,
                        'chat_id': chat_id,
                        'timestamp': time.time()
                    }

                    return True
                else:
                    logger.error(f"❌ Telegram photo failed for {camera_name}")
                    return False
            else:
                # Send text-only alert if no image
                success = self._send_telegram_message(bot_token, chat_id, message)
                if success:
                    logger.info(f"✅ Telegram text alert sent for {camera_name}")
                    return True
                else:
                    logger.error(f"❌ Telegram text failed for {camera_name}")
                    return False

        except Exception as e:
            logger.error(f"❌ Telegram alert error: {str(e)}")
            return False

    def _send_telegram_photo(self, bot_token: str, chat_id: str, photo_path: str, caption: str) -> bool:
        """Send photo to Telegram with optimized handling"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                }

                response = requests.post(url, files=files, data=data, timeout=15)

                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"❌ Telegram photo API error: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"❌ Telegram photo send error: {str(e)}")
            return False

    def _send_telegram_message(self, bot_token: str, chat_id: str, message: str) -> bool:
        """Send text message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=data, timeout=10)

            if response.status_code == 200:
                return True
            else:
                logger.error(f"❌ Telegram message API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Telegram message send error: {str(e)}")
            return False

    def send_telegram_video(self, video_path: str, camera_name: str, alert_id: str):
        """Send video to Telegram as follow-up to photo alert"""
        try:
            if not hasattr(self, 'telegram_pending_videos'):
                return

            # Find matching pending video request
            matching_alert = None
            for alert_key, alert_info in self.telegram_pending_videos.items():
                if alert_info['camera_name'] == camera_name:
                    # Check if alert is recent (within 5 minutes)
                    if time.time() - alert_info['timestamp'] < 300:
                        matching_alert = alert_info
                        break

            if not matching_alert:
                logger.debug(f"No matching Telegram alert found for video: {camera_name}")
                return

            bot_token = matching_alert['bot_token']
            chat_id = matching_alert['chat_id']

            # Check file size (Telegram limit is 50MB for bots)
            if not os.path.exists(video_path):
                logger.warning(f"⚠️ Video file not found: {video_path}")
                return

            # Check if video sending is enabled
            config = ALERT_CONFIG['telegram_bot']
            if not config.get('send_video', True):
                logger.debug(f"Telegram video sending disabled in config")
                return

            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            max_size = config.get('video_max_size_mb', 48)
            if file_size > max_size:
                logger.warning(f"⚠️ Video too large for Telegram: {file_size:.1f}MB (max: {max_size}MB)")
                return

            # Send video
            success = self._send_telegram_video_file(bot_token, chat_id, video_path, camera_name, alert_id)

            if success:
                logger.info(f"✅ Telegram video sent for {camera_name}")
                # Clean up pending video request
                self.telegram_pending_videos = {k: v for k, v in self.telegram_pending_videos.items()
                                              if v['camera_name'] != camera_name}
            else:
                logger.error(f"❌ Telegram video failed for {camera_name}")

        except Exception as e:
            logger.error(f"❌ Telegram video error: {str(e)}")

    def _send_telegram_video_file(self, bot_token: str, chat_id: str, video_path: str,
                                 camera_name: str, alert_id: str) -> bool:
        """Send video file to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendVideo"

#             caption = f"""📹 *Security Recording*

# 📹 *Camera:* {camera_name}
# 🆔 *Alert ID:* {alert_id}
# 📅 *Time:* {get_wib_timestamp()} WIB
# ⏱️ *Duration:* {SECURITY_CONFIG['recording_duration']} seconds

# 🏠 _Professional Home Security System_"""

            with open(video_path, 'rb') as video:
                files = {'video': video}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown',
                    'supports_streaming': True
                }

                # Configurable timeout for video upload
                timeout = ALERT_CONFIG['telegram_bot'].get('video_timeout', 60)
                response = requests.post(url, files=files, data=data, timeout=timeout)

                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"❌ Telegram video API error: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"❌ Telegram video upload error: {str(e)}")
            return False

    def arm_system(self):
        """Arm the security system"""
        self.armed = True
        logger.info("Security system ARMED")

    def disarm_system(self):
        """Disarm the security system"""
        self.armed = False
        logger.info("Security system DISARMED")

    def toggle_armed(self) -> bool:
        """Toggle armed status"""
        if self.armed:
            self.disarm_system()
        else:
            self.arm_system()
        return self.armed

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_time = (get_wib_time() - timedelta(hours=hours)).isoformat()

            cursor.execute('''
                SELECT * FROM security_events
                WHERE timestamp > ? AND event_type = 'detection_alert'
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            results = cursor.fetchall()
            conn.close()

            return [dict(zip([col[0] for col in cursor.description], row)) for row in results]

        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return []

    def cleanup_old_files(self):
        """Cleanup old recordings and screenshots"""
        try:
            cutoff_date = get_wib_time() - timedelta(days=RECORDING_CONFIG['auto_cleanup_days'])

            # Cleanup recordings
            for filename in os.listdir(DISPLAY_CONFIG['recordings_dir']):
                file_path = os.path.join(DISPLAY_CONFIG['recordings_dir'], filename)
                if os.path.getctime(file_path) < cutoff_date.timestamp():
                    os.remove(file_path)
                    logger.info(f"Cleaned up old recording: {filename}")

            # Cleanup old alert screenshots
            for filename in os.listdir(DISPLAY_CONFIG['alerts_dir']):
                file_path = os.path.join(DISPLAY_CONFIG['alerts_dir'], filename)
                if os.path.getctime(file_path) < cutoff_date.timestamp():
                    os.remove(file_path)
                    logger.info(f"Cleaned up old alert: {filename}")

        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    def is_armed(self) -> bool:
        """Check if system is armed"""
        return self.armed

    def stop_all_recordings(self):
        """Stop all active recordings"""
        for camera_id in list(self.active_recordings.keys()):
            self.stop_recording(camera_id)

    def shutdown(self):
        """Shutdown security manager and cleanup threads"""
        logger.info("🔒 Shutting down Security Manager...")

        # Stop cleanup thread
        self.cleanup_running = False

        # Stop all recordings
        self.stop_all_recordings()

        logger.info("✅ Security Manager shutdown complete")
