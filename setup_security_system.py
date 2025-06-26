#!/usr/bin/env python3
"""
Professional Home Security System - Auto Setup Script
Automatic configuration and installation helper
"""

import os
import sys
import subprocess
import json
import logging
from typing import Dict, List
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecuritySystemSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_executable = sys.executable
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        print("🔒 Professional Home Security System - Auto Setup")
        print("=" * 60)

    def check_python_version(self):
        """Check Python version compatibility"""
        print("📋 Checking Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} not supported")
            print("✅ Please install Python 3.8 or higher")
            return False

        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

    def install_dependencies(self):
        """Install required Python packages"""
        print("\n📦 Installing dependencies...")

        try:
            # Upgrade pip first
            subprocess.run([self.python_executable, "-m", "pip", "install", "--upgrade", "pip"],
                          check=True, capture_output=True)

            # Install requirements
            requirements_file = os.path.join(self.project_dir, "requirements.txt")
            if os.path.exists(requirements_file):
                subprocess.run([self.python_executable, "-m", "pip", "install", "-r", requirements_file],
                              check=True)
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ requirements.txt not found")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    def check_gpu_support(self):
        """Check for GPU support"""
        print("\n🎮 Checking GPU support...")

        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ CUDA GPU detected: {gpu_name} ({gpu_count} device(s))")
                print("🚀 GPU acceleration will be enabled")
                return True
            else:
                print("⚠️ No CUDA GPU detected - CPU mode will be used")
                print("💡 For better performance, consider installing CUDA")
                return False
        except ImportError:
            print("⚠️ PyTorch not installed - installing CPU version")
            return False

    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")

        directories = [
            "screenshots",
            "alerts",
            "recordings",
            "logs",
            "assets"
        ]

        for directory in directories:
            dir_path = os.path.join(self.project_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created: {directory}/")

        return True

    def setup_camera_config(self):
        """Interactive camera configuration"""
        print("\n📹 Camera Configuration")
        print("-" * 30)

        cameras = []
        camera_names = ["Pintu Depan", "Garasi", "Gerbang", "Halaman"]
        zones = ["entry", "perimeter", "perimeter", "outdoor"]
        priorities = ["critical", "high", "critical", "medium"]

        # Ask for IP address
        while True:
            ip = input("Enter camera/NVR IP address (e.g., 192.168.1.108): ").strip()
            if ip:
                break
            print("❌ Please enter a valid IP address")

        # Ask for credentials
        username = input("Enter username (default: admin): ").strip() or "admin"
        password = input("Enter password (default: admin123): ").strip() or "admin123"

        # Configure cameras
        for i in range(4):
            print(f"\n📷 Configuring Camera {i+1}: {camera_names[i]}")

            # Ask if user wants to configure this camera
            configure = input(f"Configure {camera_names[i]}? (y/n, default: y): ").strip().lower()
            if configure in ['n', 'no']:
                continue

            camera_config = {
                'id': str(i + 1),
                'ip': ip,
                'username': username,
                'password': password,
                'channel': i + 1,
                'name': camera_names[i],
                'security_zone': zones[i],
                'priority': priorities[i],
                'record_alerts': True
            }

            cameras.append(camera_config)
            print(f"✅ {camera_names[i]} configured (Channel {i+1}, Zone: {zones[i]})")

        return cameras

    def setup_alert_config(self):
        """Configure alert system"""
        print("\n🚨 Alert System Configuration")
        print("-" * 30)

        alert_config = {
            'email_alerts': {'enabled': False},
            'sound_alerts': {'enabled': True, 'alert_sound': 'assets/alert.wav', 'volume': 0.8},
            'telegram_bot': {'enabled': False}
        }

        # Email configuration
        email_setup = input("Setup email alerts? (y/n, default: n): ").strip().lower()
        if email_setup in ['y', 'yes']:
            print("\n📧 Email Alert Configuration:")
            email = input("Enter your Gmail address: ").strip()
            password = input("Enter your Gmail App Password: ").strip()
            recipient = input("Enter alert recipient email: ").strip()

            if email and password and recipient:
                alert_config['email_alerts'] = {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'email': email,
                    'password': password,
                    'recipients': [recipient]
                }
                print("✅ Email alerts configured")
            else:
                print("⚠️ Email configuration skipped - incomplete information")

        # Telegram configuration
        telegram_setup = input("\nSetup Telegram alerts? (y/n, default: n): ").strip().lower()
        if telegram_setup in ['y', 'yes']:
            print("\n📱 Telegram Bot Configuration:")
            bot_token = input("Enter bot token (from @BotFather): ").strip()
            chat_id = input("Enter your chat ID: ").strip()

            if bot_token and chat_id:
                # Additional Telegram options
                send_video = input("Send video recordings to Telegram? (y/n, default: y): ").strip().lower()
                send_video = send_video not in ['n', 'no']

                alert_config['telegram_bot'] = {
                    'enabled': True,
                    'bot_token': bot_token,
                    'chat_id': chat_id,
                    'send_photo': True,
                    'send_video': send_video,
                    'video_max_size_mb': 48,
                    'video_timeout': 60,
                    'enhanced_formatting': True
                }
                print("✅ Telegram alerts configured")
                if send_video:
                    print("   📹 Video recordings will be sent to Telegram")
                else:
                    print("   📸 Only photos will be sent to Telegram")
            else:
                print("⚠️ Telegram configuration skipped - incomplete information")

        return alert_config

    def generate_config_file(self, cameras: List[Dict], alerts: Dict):
        """Generate config.py file"""
        print("\n⚙️ Generating configuration file...")

        config_content = f'''# Professional Home Security System Configuration
# Auto-generated by setup script

# Camera Configuration
CAMERAS_CONFIG = {json.dumps(cameras, indent=4)}

# YOLO Detection Configuration - Optimized for Performance
YOLO_CONFIG = {{
    'model_path': 'yolov8n.pt',  # Fastest model for real-time
    'confidence_threshold': 0.4,  # Slightly lower for better detection
    'detection_classes': ['person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'cat', 'dog'],
    'alert_classes': ['person', 'car', 'motorcycle'],
    'detection_enabled': True,
    'show_labels': True,
    'show_confidence': True,
    'detection_interval': 3,  # Detect every 3 frames for FPS optimization
    'async_detection': True,  # Async processing
    'max_detections': 10,  # Limit detections per frame
    'nms_threshold': 0.45,  # Non-maximum suppression
}}

# Performance Configuration
PERFORMANCE_CONFIG = {{
    'target_fps': 15,  # Target FPS for smooth playback
    'detection_fps': 5,  # FPS for YOLO detection (lower)
    'frame_skip_detection': 3,  # Skip frames for detection
    'buffer_size': 1,
    'thread_pool_size': 4,
    'gpu_acceleration': True,
    'memory_optimization': True
}}

# Security Configuration
SECURITY_CONFIG = {{
    'armed': False,  # Start disarmed for initial setup
    'sensitivity': 'high',  # low, medium, high, critical
    'alert_cooldown': 30,  # Seconds between same-type alerts
    'recording_duration': 30,  # Seconds to record after detection
    'backup_alerts': True,
    'continuous_recording': False,
    'motion_sensitivity': 0.3,
    'intrusion_zones': ['entry', 'perimeter'],
    'night_mode': False,
    'auto_arm_schedule': {{
        'enabled': False,  # Enable after testing
        'arm_time': '22:00',
        'disarm_time': '06:00'
    }}
}}

# Alert Configuration
ALERT_CONFIG = {json.dumps(alerts, indent=4)}

# Display Configuration - 16:9 Optimized
DISPLAY_CONFIG = {{
    'window_width': 1280,
    'window_height': 720,
    'grid_size_2x2': (640, 360),   # Perfect 16:9
    'grid_size_1x4': (320, 180),   # Perfect 16:9
    'grid_size_4x1': (320, 180),   # Perfect 16:9
    'grid_size_1x1': (1280, 720),  # Perfect 16:9
    'screenshot_dir': 'screenshots',
    'alerts_dir': 'alerts',
    'recordings_dir': 'recordings',
    'logs_dir': 'logs',
    'full_hd_screenshots': True,  # 1920x1080 screenshots
    'compression_quality': 95
}}

# Recording Configuration
RECORDING_CONFIG = {{
    'enabled': True,
    'format': 'mp4',
    'codec': 'h264',
    'fps': 15,
    'resolution': (1280, 720),  # 16:9 HD recording
    'max_file_size_mb': 100,
    'auto_cleanup_days': 30,
    'backup_to_cloud': False
}}

# Colors for bounding boxes (BGR format) - Enhanced
DETECTION_COLORS = {{
    'person': (0, 255, 0),       # Green - Friendly
    'car': (255, 0, 0),          # Blue - Vehicle
    'motorcycle': (0, 0, 255),   # Red - High alert
    'bicycle': (255, 255, 0),    # Cyan - Low priority
    'bus': (255, 0, 255),        # Magenta
    'truck': (0, 255, 255),      # Yellow
    'cat': (128, 255, 128),      # Light green
    'dog': (128, 255, 128)       # Light green
}}

# Zone Colors for Security Areas
ZONE_COLORS = {{
    'entry': (0, 0, 255),        # Red - Critical entry points
    'perimeter': (255, 165, 0),  # Orange - Perimeter security
    'outdoor': (0, 255, 255),    # Yellow - General outdoor
    'indoor': (0, 255, 0)        # Green - Indoor monitoring
}}
'''

        config_file = os.path.join(self.project_dir, "config.py")
        with open(config_file, 'w') as f:
            f.write(config_content)

        print("✅ Configuration file generated: config.py")
        return True

    def test_system(self):
        """Test system components"""
        print("\n🔧 Testing system components...")

        try:
            # Test imports
            print("Testing imports...")
            import cv2
            print(f"✅ OpenCV {cv2.__version__}")

            import numpy as np
            print(f"✅ NumPy {np.__version__}")

            try:
                from ultralytics import YOLO
                print("✅ Ultralytics YOLO")
            except ImportError:
                print("⚠️ Ultralytics not available - installing...")
                subprocess.run([self.python_executable, "-m", "pip", "install", "ultralytics"], check=True)
                print("✅ Ultralytics installed")

            # Test YOLO model download
            print("Testing YOLO model...")
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # This will download if needed
            print("✅ YOLOv8n model ready")

            return True

        except Exception as e:
            print(f"❌ System test failed: {e}")
            return False

    def run_setup(self):
        """Run complete setup process"""
        print("Starting Professional Home Security System setup...\n")

        # Step 1: Check Python version
        if not self.check_python_version():
            return False

        # Step 2: Install dependencies
        if not self.install_dependencies():
            return False

        # Step 3: Check GPU support
        self.check_gpu_support()

        # Step 4: Create directories
        if not self.create_directories():
            return False

        # Step 5: Configure cameras
        cameras = self.setup_camera_config()
        if not cameras:
            print("❌ No cameras configured")
            return False

        # Step 6: Configure alerts
        alerts = self.setup_alert_config()

        # Step 7: Generate config file
        if not self.generate_config_file(cameras, alerts):
            return False

        # Step 8: Test system
        if not self.test_system():
            print("⚠️ System test failed - but setup completed")

        # Final success message
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Dependencies installed")
        print(f"✅ {len(cameras)} camera(s) configured")
        print("✅ Alert system configured")
        print("✅ Configuration file generated")
        print("✅ Directories created")
        print("\n🚀 To start the security system:")
        print("   python main.py")
        print("\n📖 For more information:")
        print("   - Read README.md")
        print("   - Check config.py for advanced settings")
        print("   - Visit logs/ folder for system logs")
        print("\n🔒 Your home security system is ready!")

        return True

def main():
    """Main setup function"""
    try:
        setup = SecuritySystemSetup()
        success = setup.run_setup()

        if success:
            sys.exit(0)
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
