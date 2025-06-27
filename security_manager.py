import cv2
import numpy as np
import threading
import time
import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
import hashlib

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import (
    SECURITY_CONFIG, ALERT_CONFIG, RECORDING_CONFIG,
    get_wib_timestamp, get_wib_filename_timestamp,
    should_send_alert, YOLO_CONFIG
)

logger = logging.getLogger(__name__)

class AlertThrottler:
    def __init__(self):
        self.alert_counts = {'hour': {}, 'day': {}, 'last_times': {}}
        self.cross_camera_alerts = {}
        self.config = YOLO_CONFIG['anti_spam']

        self.dedup_config = self.config.get('cross_camera_dedup', {
            'enabled': True,
            'time_window': 45,
            'similar_cameras': {}
        })

    def should_allow_alert(self, camera_name: str, detection_type: str) -> bool:
        if not self.config['enabled']:
            return True

        if self.dedup_config['enabled']:
            if self._is_duplicate_cross_camera(camera_name, detection_type):
                logger.debug(f"🔄 Cross-camera duplicate: {camera_name} - {detection_type}")
                return False

        key = f"{camera_name}_{detection_type}"
        current_time = time.time()

        # Cek cooldown
        if key in self.alert_counts.get('last_times', {}):
            if current_time - self.alert_counts['last_times'][key] < self.config['alert_spam_cooldown']:
                return False

        # Cek limit per jam
        hour_key = int(current_time // 3600)
        if hour_key not in self.alert_counts['hour']:
            self.alert_counts['hour'][hour_key] = {}
        if key not in self.alert_counts['hour'][hour_key]:
            self.alert_counts['hour'][hour_key][key] = 0

        if self.alert_counts['hour'][hour_key][key] >= self.config['max_alerts_per_hour']:
            return False

        # Cek limit per hari
        day_key = int(current_time // 86400)
        if day_key not in self.alert_counts['day']:
            self.alert_counts['day'][day_key] = {}
        if key not in self.alert_counts['day'][day_key]:
            self.alert_counts['day'][day_key][key] = 0

        if self.alert_counts['day'][day_key][key] >= self.config['max_alerts_per_day']:
            return False

        self.alert_counts['hour'][hour_key][key] += 1
        self.alert_counts['day'][day_key][key] += 1
        self.alert_counts['last_times'][key] = current_time

        if self.dedup_config['enabled']:
            self._track_cross_camera_alert(camera_name, detection_type, current_time)

        self._cleanup_old_entries()
        return True

    def _cleanup_old_entries(self):
        current_time = time.time()
        current_hour = int(current_time // 3600)
        current_day = int(current_time // 86400)

        old_hours = [h for h in self.alert_counts['hour'].keys() if h < current_hour - 2]
        for h in old_hours:
            del self.alert_counts['hour'][h]

        old_days = [d for d in self.alert_counts['day'].keys() if d < current_day - 2]
        for d in old_days:
            del self.alert_counts['day'][d]

    def _is_duplicate_cross_camera(self, camera_name: str, detection_type: str) -> bool:
        current_time = time.time()
        time_window = self.dedup_config['time_window']

        camera_zone = None
        for zone, cameras in self.dedup_config['similar_cameras'].items():
            if camera_name in cameras:
                camera_zone = zone
                break

        if not camera_zone:
            return False

        zone_key = f"{camera_zone}_{detection_type}"

        if zone_key in self.cross_camera_alerts:
            last_alert_time = self.cross_camera_alerts[zone_key]['time']
            last_camera = self.cross_camera_alerts[zone_key]['camera']

            if (current_time - last_alert_time) <= time_window and last_camera != camera_name:
                logger.info(f"🔄 Duplicate detected: {camera_name} (sama seperti {last_camera} {int(current_time - last_alert_time)}s lalu)")
                return True

        return False

    def _track_cross_camera_alert(self, camera_name: str, detection_type: str, timestamp: float):
        camera_zone = None
        for zone, cameras in self.dedup_config['similar_cameras'].items():
            if camera_name in cameras:
                camera_zone = zone
                break

        if camera_zone:
            zone_key = f"{camera_zone}_{detection_type}"
            self.cross_camera_alerts[zone_key] = {
                'camera': camera_name,
                'time': timestamp,
                'detection_type': detection_type
            }

            cleanup_time = timestamp - (self.dedup_config['time_window'] * 2)
            to_remove = []
            for key, data in self.cross_camera_alerts.items():
                if data['time'] < cleanup_time:
                    to_remove.append(key)

            for key in to_remove:
                del self.cross_camera_alerts[key]

class NotificationSender:
    def __init__(self):
        self.telegram_config = ALERT_CONFIG.get('telegram_bot', {})
        self.webhook_config = ALERT_CONFIG.get('webhook_alerts', {})
        self.throttler = AlertThrottler()

    def send_alert(self, camera_name: str, detection_type: str, confidence: float,
                   image_path: str, zone: str, priority: str = "medium") -> bool:

        if not should_send_alert(detection_type, zone):
            logger.debug(f"Skip alert untuk {detection_type} di zone {zone}")
            return False

        if not self.throttler.should_allow_alert(camera_name, detection_type):
            logger.debug(f"Alert di-throttle untuk {camera_name} - {detection_type}")
            return False

        success = False

        if self.telegram_config.get('enabled', False):
            success |= self._send_telegram_alert(camera_name, detection_type, confidence, image_path, priority)

        if self.webhook_config.get('enabled', False):
            discord_config = self.webhook_config.get('services', {}).get('discord', {})
            if discord_config.get('enabled', False):
                success |= self._send_discord_alert(camera_name, detection_type, confidence, image_path, priority)

        return success

    def _send_telegram_alert(self, camera_name: str, detection_type: str, confidence: float,
                           image_path: str, priority: str) -> bool:
        try:
            bot_token = self.telegram_config.get('bot_token', '')
            chat_id = self.telegram_config.get('chat_id', '')

            if not bot_token or not chat_id:
                return False

            message = f"🚨 *SECURITY ALERT*\n\n"
            message += f"📍 *Camera:* {camera_name}\n"
            message += f"🎯 *Detection:* {detection_type.upper()}\n"
            message += f"📊 *Confidence:* {confidence:.1%}\n"
            message += f"⚠️ *Priority:* {priority.upper()}\n"
            message += f"🕒 *Time (WIB):* {get_wib_timestamp()}"

            if self.telegram_config.get('send_photo', True) and os.path.exists(image_path):
                return self._send_telegram_photo(bot_token, chat_id, image_path, message)
            else:
                return self._send_telegram_message(bot_token, chat_id, message)

        except Exception as e:
            logger.error(f"❌ Telegram alert error: {e}")
            return False

    def _send_telegram_photo(self, bot_token: str, chat_id: str, photo_path: str, caption: str) -> bool:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': chat_id, 'caption': caption, 'parse_mode': 'Markdown'}
                response = requests.post(url, files=files, data=data, timeout=30)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Telegram photo error: {e}")
            return False

    def _send_telegram_message(self, bot_token: str, chat_id: str, message: str) -> bool:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Telegram message error: {e}")
            return False

    def _send_discord_alert(self, camera_name: str, detection_type: str, confidence: float,
                          image_path: str, priority: str) -> bool:
        """Kirim alert ke Discord"""
        try:
            services = self.webhook_config.get('services', {})
            discord_config = services.get('discord', {})
            webhook_url = discord_config.get('webhook_url', '')

            if not webhook_url:
                return False

            # Warna berdasarkan priority
            colors = {'critical': 0xff0000, 'high': 0xff8800, 'medium': 0xffff00, 'low': 0x00ff00}

            # Buat embed
            embed = {
                "title": "🚨 Security Alert",
                "description": f"**{detection_type.upper()}** detected at **{camera_name}**",
                "color": colors.get(priority, 0xff0000),
                "fields": [
                    {"name": "📍 Location", "value": camera_name, "inline": True},
                    {"name": "🎯 Detection", "value": f"{detection_type} ({confidence:.1%})", "inline": True},
                    {"name": "⚠️ Priority", "value": priority.upper(), "inline": True},
                    {"name": "🕒 Time (WIB)", "value": get_wib_timestamp(), "inline": False}
                ],
                "footer": {"text": "Security System"},
                "timestamp": get_wib_timestamp("%Y-%m-%dT%H:%M:%S+07:00")
            }

            payload = {
                "username": discord_config.get('username', 'Security System'),
                "avatar_url": discord_config.get('avatar_url', ''),
                "embeds": [embed]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            success = response.status_code in [200, 204]

            # Kirim gambar jika diaktifkan
            if success and discord_config.get('send_image', True) and os.path.exists(image_path):
                self._send_discord_image(webhook_url, discord_config, image_path)

            return success

        except Exception as e:
            logger.error(f"❌ Discord alert error: {e}")
            return False

    def _send_discord_image(self, webhook_url: str, config: dict, image_path: str):
        """Kirim gambar ke Discord sebagai follow-up"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (f'alert_{int(time.time())}.jpg', f, 'image/jpeg')}
                payload = {"username": config.get('username', 'Security System')}
                requests.post(webhook_url, data=payload, files=files, timeout=10)
        except Exception as e:
            logger.error(f"❌ Discord image error: {e}")

    def send_video(self, video_path: str, camera_name: str, alert_id: str):
        """Kirim video ke service yang mengaktifkan video"""
        if not os.path.exists(video_path):
            return

        # Kirim ke Telegram jika video diaktifkan
        if (self.telegram_config.get('enabled', False) and
            self.telegram_config.get('send_video', False)):
            self._send_telegram_video(video_path, camera_name, alert_id)

        # Kirim ke Discord jika video diaktifkan
        discord_config = self.webhook_config.get('services', {}).get('discord', {})
        if (discord_config.get('enabled', False) and
            discord_config.get('send_video', False)):
            self._send_discord_video(video_path, camera_name, alert_id)

    def _send_telegram_video(self, video_path: str, camera_name: str, alert_id: str):
        """Kirim video ke Telegram"""
        try:
            bot_token = self.telegram_config.get('bot_token', '')
            chat_id = self.telegram_config.get('chat_id', '')

            if not bot_token or not chat_id:
                return

            url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
            with open(video_path, 'rb') as video:
                files = {'video': video}
                data = {
                    'chat_id': chat_id,
                    'caption': f"📹 Security Recording from {camera_name} (Alert ID: {alert_id})",
                    'supports_streaming': True
                }
                response = requests.post(url, files=files, data=data, timeout=60)
                if response.status_code == 200:
                    logger.info(f"✅ Telegram video sent for {camera_name}")

        except Exception as e:
            logger.error(f"❌ Telegram video error: {e}")

    def _send_discord_video(self, video_path: str, camera_name: str, alert_id: str):
        """Kirim video ke Discord"""
        try:
            # Discord maksimal 25MB, cek ukuran dulu
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            if file_size_mb > 24:  # Beri margin
                logger.warning(f"Discord video terlalu besar: {file_size_mb:.1f}MB")
                return

            services = self.webhook_config.get('services', {})
            discord_config = services.get('discord', {})
            webhook_url = discord_config.get('webhook_url', '')

            if not webhook_url:
                return

            with open(video_path, 'rb') as f:
                files = {'file': (f'recording_{alert_id}.mp4', f, 'video/mp4')}
                payload = {
                    "username": discord_config.get('username', 'Security System'),
                    "content": f"📹 Security Recording from {camera_name} (Alert ID: {alert_id})"
                }
                response = requests.post(webhook_url, data=payload, files=files, timeout=60)
                if response.status_code in [200, 204]:
                    logger.info(f"✅ Discord video sent for {camera_name}")

        except Exception as e:
            logger.error(f"❌ Discord video error: {e}")

class SecurityManager:
    """Class utama untuk mengelola sistem keamanan"""

    def __init__(self):
        self.armed = SECURITY_CONFIG.get('armed', True)
        self.active_recordings = {}
        self.notification_sender = NotificationSender()

        # Initialize database
        self.init_database()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info("🛡️ Security Manager initialized")

    def _cleanup_loop(self):
        """Background cleanup file dan database entries lama"""
        while True:
            try:
                self.cleanup_old_files()
                time.sleep(3600)  # Jalankan setiap jam
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")
                time.sleep(300)  # Retry dalam 5 menit

    def init_database(self):
        """Initialize SQLite database untuk security events"""
        try:
            os.makedirs('logs', exist_ok=True)
            conn = sqlite3.connect('logs/security.db')
            cursor = conn.cursor()

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

            conn.commit()
            conn.close()
            logger.info("📄 Security database initialized")

        except Exception as e:
            logger.error(f"❌ Database init error: {e}")

    def log_event(self, camera_id: str, camera_name: str, event_type: str,
                  detected_objects: List = None, confidence: float = 0.0,
                  priority: str = "medium", zone: str = "unknown",
                  alert_sent: bool = False, screenshot_path: str = None,
                  recording_path: str = None):
        """Log security event ke database"""
        try:
            conn = sqlite3.connect('logs/security.db')
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO security_events
                (timestamp, camera_id, camera_name, event_type, detected_objects,
                 confidence, priority, zone, alert_sent, screenshot_path, recording_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                get_wib_timestamp(), camera_id, camera_name, event_type,
                str(detected_objects) if detected_objects else None,
                confidence, priority, zone, alert_sent, screenshot_path, recording_path
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ Database log error: {e}")

    def process_detection(self, camera_id: str, camera_name: str, detections: List,
                         frame: np.ndarray, camera_config: Dict) -> bool:
        """Proses YOLO detections dan trigger alerts jika diperlukan"""
        if not self.armed or not detections:
            return False

        zone = camera_config.get('security_zone', 'unknown')
        priority = camera_config.get('priority', 'medium')

        # Filter hanya deteksi yang layak untuk alert
        alert_detections = []
        for detection in detections:
            detection_type = detection.get('class', 'unknown')
            if should_send_alert(detection_type, zone):
                alert_detections.append(detection)

        if not alert_detections:
            return False

        # Generate alert ID
        alert_id = hashlib.md5(f"{camera_name}_{int(time.time())}".encode()).hexdigest()[:8]

        # Simpan screenshot
        screenshot_path = self.save_alert_screenshot(frame, camera_name, alert_detections, alert_id)

        # Mulai recording jika diaktifkan
        recording_path = None
        if camera_config.get('record_alerts', True):
            recording_path = self.start_alert_recording(camera_id, camera_name, alert_id)

        # Kirim alerts
        alert_sent = False
        for detection in alert_detections:
            detection_type = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)

            success = self.notification_sender.send_alert(
                camera_name, detection_type, confidence,
                screenshot_path, zone, priority
            )

            if success:
                alert_sent = True
                logger.info(f"✅ Alert sent for {detection_type} at {camera_name}")

        # Main sound alert
        if alert_sent and ALERT_CONFIG.get('sound_alerts', {}).get('enabled', False):
            self.play_alert_sound()

        # Log event
        self.log_event(
            camera_id, camera_name, 'detection',
            [d.get('class') for d in alert_detections],
            max([d.get('confidence', 0) for d in alert_detections]),
            priority, zone, alert_sent, screenshot_path, recording_path
        )

        return alert_sent

    def save_alert_screenshot(self, frame: np.ndarray, camera_name: str,
                            detections: List, alert_id: str) -> str:
        """Simpan screenshot alert dengan overlay deteksi"""
        try:
            os.makedirs('alerts', exist_ok=True)

            # Buat screenshot profesional (1920x1080)
            screenshot = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

            # Tambah overlay deteksi
            for detection in detections:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    # Scale bbox ke resolusi screenshot
                    x1 = int(bbox[0] * 1920 / frame.shape[1])
                    y1 = int(bbox[1] * 1080 / frame.shape[0])
                    x2 = int(bbox[2] * 1920 / frame.shape[1])
                    y2 = int(bbox[3] * 1080 / frame.shape[0])

                    # Gambar detection box
                    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Tambah label
                    label = f"{detection.get('class', 'Unknown')} {detection.get('confidence', 0):.1%}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(screenshot, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (0, 0, 255), -1)
                    cv2.putText(screenshot, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Tambah timestamp dan info kamera
            timestamp = get_wib_timestamp()
            info_text = f"{camera_name} - {timestamp} WIB - Alert ID: {alert_id}"
            cv2.putText(screenshot, info_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Simpan screenshot
            filename = f"ALERT_{camera_name.replace(' ', '_')}_{alert_id}_{get_wib_filename_timestamp()}.jpg"
            filepath = os.path.join('alerts', filename)
            cv2.imwrite(filepath, screenshot, [cv2.IMWRITE_JPEG_QUALITY, 95])

            return filepath

        except Exception as e:
            logger.error(f"❌ Screenshot save error: {e}")
            return None

    def start_alert_recording(self, camera_id: str, camera_name: str, alert_id: str) -> str:
        """Mulai recording untuk alert"""
        try:
            os.makedirs('recordings', exist_ok=True)

            # Stop recording yang ada untuk kamera ini
            if camera_id in self.active_recordings:
                self.stop_recording(camera_id)

            # Buat filename recording
            filename = f"REC_{camera_name.replace(' ', '_')}_{alert_id}_{get_wib_filename_timestamp()}.mp4"
            filepath = os.path.join('recordings', filename)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = RECORDING_CONFIG.get('fps', 15)

            self.active_recordings[camera_id] = {
                'writer': cv2.VideoWriter(filepath, fourcc, fps, (640, 480)),
                'filepath': filepath,
                'start_time': time.time(),
                'alert_id': alert_id,
                'camera_name': camera_name
            }

            logger.info(f"📹 Recording started for {camera_name}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Recording start error: {e}")
            return None

    def update_recording(self, camera_id: str, frame: np.ndarray):
        """Update active recording dengan frame baru"""
        if camera_id not in self.active_recordings:
            return

        try:
            recording = self.active_recordings[camera_id]
            writer = recording['writer']

            # Resize frame untuk recording
            recording_frame = cv2.resize(frame, (640, 480))
            writer.write(recording_frame)

            # Cek apakah recording harus dihentikan
            duration = time.time() - recording['start_time']
            max_duration = SECURITY_CONFIG.get('recording_duration', 20)

            if duration >= max_duration:
                self.stop_recording(camera_id)

        except Exception as e:
            logger.error(f"❌ Recording update error: {e}")

    def stop_recording(self, camera_id: str):
        """Stop recording untuk kamera"""
        if camera_id not in self.active_recordings:
            return

        try:
            recording = self.active_recordings[camera_id]
            recording['writer'].release()

            filepath = recording['filepath']
            alert_id = recording['alert_id']
            camera_name = recording['camera_name']

            del self.active_recordings[camera_id]

            # Kirim video jika diaktifkan
            if os.path.exists(filepath):
                threading.Thread(
                    target=self.notification_sender.send_video,
                    args=(filepath, camera_name, alert_id),
                    daemon=True
                ).start()

            logger.info(f"📹 Recording stopped for {camera_name}")

        except Exception as e:
            logger.error(f"❌ Recording stop error: {e}")

    def play_alert_sound(self):
        """Main alert sound"""
        if not PYGAME_AVAILABLE:
            return

        try:
            sound_config = ALERT_CONFIG['sound_alerts']
            sound_file = sound_config.get('alert_sound', 'assets/alert.wav')

            if os.path.exists(sound_file):
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.set_volume(sound_config.get('volume', 0.6))
                pygame.mixer.music.play()

        except Exception as e:
            logger.error(f"❌ Sound alert error: {e}")

    def cleanup_old_files(self):
        """Bersihkan file alert dan recording lama"""
        try:
            cleanup_days = RECORDING_CONFIG.get('cleanup_days', 7)
            cutoff_time = time.time() - (cleanup_days * 24 * 3600)
            files_cleaned = 0

            # Bersihkan alerts dan recordings
            for dirname in ['alerts', 'recordings']:
                if not os.path.exists(dirname):
                    continue

                for filename in os.listdir(dirname):
                    filepath = os.path.join(dirname, filename)
                    try:
                        if os.path.getctime(filepath) < cutoff_time:
                            os.remove(filepath)
                            files_cleaned += 1
                    except OSError:
                        continue

            # Bersihkan database entries lama
            with sqlite3.connect('logs/security.db') as conn:
                cursor = conn.cursor()
                cutoff_date = datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('DELETE FROM security_events WHERE timestamp < ?', (cutoff_date,))
                deleted_entries = cursor.rowcount

            if files_cleaned > 0 or deleted_entries > 0:
                logger.info(f"🗑️ Cleanup: {files_cleaned} files, {deleted_entries} DB entries (>{cleanup_days}d)")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

    def arm_system(self):
        """Aktifkan sistem keamanan"""
        self.armed = True
        logger.info("🛡️ Security system ARMED")

    def disarm_system(self):
        """Nonaktifkan sistem keamanan"""
        self.armed = False
        logger.info("🔓 Security system DISARMED")

    def toggle_armed(self) -> bool:
        """Toggle status armed"""
        if self.armed:
            self.disarm_system()
        else:
            self.arm_system()
        return self.armed

    def is_armed(self) -> bool:
        """Cek apakah sistem aktif"""
        return self.armed

    def stop_all_recordings(self):
        """Stop semua recording yang aktif"""
        for camera_id in list(self.active_recordings.keys()):
            self.stop_recording(camera_id)

    def shutdown(self):
        """Shutdown security manager"""
        self.stop_all_recordings()
        logger.info("🔴 Security Manager shutdown")
