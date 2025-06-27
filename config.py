import os
import pytz
from dotenv import load_dotenv

load_dotenv()

TIMEZONE = pytz.timezone('Asia/Jakarta')
CAMERAS_CONFIG = [
    {
        'id': '1',
        'ip': os.getenv('CAMERA_IP', '192.168.1.108'),
        'username': os.getenv('CAMERA_USERNAME', 'admin'),
        'password': os.getenv('CAMERA_PASSWORD', 'admin123'),
        'channel': 1,
        'name': 'Halaman Kost',
        'security_zone': 'entry',
        'priority': 'high',
        'record_alerts': True
    },
    {
        'id': '2',
        'ip': os.getenv('CAMERA_IP', '192.168.1.108'),
        'username': os.getenv('CAMERA_USERNAME', 'admin'),
        'password': os.getenv('CAMERA_PASSWORD', 'admin123'),
        'channel': 2,
        'name': 'Gerbang Kost',
        'security_zone': 'perimeter',
        'priority': 'high',
        'record_alerts': True
    },
    {
        'id': '3',
        'ip': os.getenv('CAMERA_IP', '192.168.1.108'),
        'username': os.getenv('CAMERA_USERNAME', 'admin'),
        'password': os.getenv('CAMERA_PASSWORD', 'admin123'),
        'channel': 3,
        'name': 'Gerbang Rumah',
        'security_zone': 'perimeter',
        'priority': 'critical',
        'record_alerts': True
    },
    {
        'id': '4',
        'ip': os.getenv('CAMERA_IP', '192.168.1.108'),
        'username': os.getenv('CAMERA_USERNAME', 'admin'),
        'password': os.getenv('CAMERA_PASSWORD', 'admin123'),
        'channel': 4,
        'name': 'Halaman Rumah',
        'security_zone': 'outdoor',
        'priority': 'critical',
        'record_alerts': True
    }
]

HARDWARE_CONFIG = {
    'use_gpu': True,
    'gpu_device': 0,
    'force_cpu': False,
    'auto_detect_best': True,
    'benchmark_on_startup': False,

    'cuda_optimizations': {
        'use_half_precision': True,
        'use_tensorrt': False,
        'memory_fraction': 0.8,
        'allow_growth': True,
    },

    'cpu_optimizations': {
        'num_threads': 4,
        'use_mkldnn': True,
        'inter_op_threads': 2,
        'intra_op_threads': 4,
    }
}

YOLO_CONFIG = {
    'model_path': 'yolov8l.pt',
    'confidence_threshold': 0.65,  # Lebih tinggi untuk mengurangi false positive
    'confidence_threshold_night': 0.75,  # Lebih tinggi untuk malam
    'confidence_threshold_24h': 0.7,  # Lebih tinggi untuk all day
    'nms_threshold': 0.45,
    'detection_enabled': True,
    'async_detection': False,
    'show_labels': True,
    'show_confidence': True,
    'auto_night_mode': True,

    'detection_classes': ['person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'cat', 'dog'],
    'alert_classes': ['person'],

    'zone_alert_classes': {
        'entry': ['person'],
        'perimeter': ['person'],
        'outdoor': ['person'],
        'indoor': ['person'],
    },

    'max_detections': 10,
    'detection_interval': 2,

    'anti_spam': {
        'enabled': True,
        'movement_threshold': 80,
        'static_object_timeout': 180,
        'alert_spam_cooldown': 120,
        'same_detection_cooldown': 300,
        'cleanup_interval': 300,
        'max_alerts_per_hour': 20,
        'max_alerts_per_day': 200,

        'cross_camera_dedup': {
            'enabled': True,
            'time_window': 45,
            'similar_cameras': {
                'gerbang': ['Gerbang Rumah', 'Gerbang Kost'],
                'halaman': ['Halaman Depan', 'Halaman Kost', 'Teras Depan'],
                'parkir': ['Parkir Mobil', 'Garasi', 'Area Parkir'],
                'entrance': ['Pintu Masuk', 'Gerbang Utama', 'Entry Point'],
                'backyard': ['Halaman Belakang', 'Taman Belakang', 'Area Belakang']
            }
        }
    },

    'gpu_settings': {
        'batch_size': 2,
        'input_size': 640,
        'warmup_iterations': 3,
    },

    'cpu_settings': {
        'batch_size': 1,
        'input_size': 640,
        'warmup_iterations': 2,
    }
}

PERFORMANCE_CONFIG = {
    'gpu_profile': {
        'target_fps': 20,
        'detection_fps': 8,
        'frame_skip_detection': 3,
        'buffer_size': 1,
        'thread_pool_size': 4,
        'memory_optimization': True,
        'prefetch_frames': True,
    },

    'cpu_profile': {
        'target_fps': 15,
        'detection_fps': 4,
        'frame_skip_detection': 4,
        'buffer_size': 1,
        'thread_pool_size': 2,
        'memory_optimization': True,
        'prefetch_frames': False,
    }
}

SECURITY_CONFIG = {
    'armed': True,
    'sensitivity': 'medium',
    'alert_cooldown': 120,
    'recording_duration': 20,
    'backup_alerts': True,
    'continuous_recording': False,
    'motion_sensitivity': 0.3,
    'night_mode': True,
    'auto_enable_detection': True,
    'force_armed_on_startup': True,

    'intrusion_zones': ['entry', 'perimeter'],
    'auto_arm_schedule': {
        'enabled': False,
        'arm_time': '22:00',
        'disarm_time': '06:00'
    }
}

# ==================== ALERT CONFIGURATION ====================
ALERT_CONFIG = {
    'email_alerts': {
        'enabled': False,
        'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
        'email': os.getenv('EMAIL_ADDRESS', 'your-email@gmail.com'),
        'password': os.getenv('EMAIL_PASSWORD', 'your-app-password'),
        'recipients': os.getenv('EMAIL_RECIPIENTS', 'security@yourdomain.com').split(','),
        'send_video': False,  # Opsi untuk mengirim video
    },

    'sound_alerts': {
        'enabled': True,
        'alert_sound': 'assets/alert.wav',
        'volume': 0.6  # Turunkan dari 0.8 ke 0.6
    },

    'telegram_bot': {
        'enabled': True,
        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
        'send_photo': True,
        'send_video': False,  # DEFAULT: Tidak mengirim video
        'video_max_size_mb': 48,
        'video_timeout': 60,
        'enhanced_formatting': True,
        'max_photos_per_hour': 10,  # Maksimal 10 foto per jam
        'max_videos_per_hour': 5,   # Maksimal 5 video per jam
    },

    'webhook_alerts': {
        'enabled': True,
        'retry_attempts': 2,  # Turunkan dari 3 ke 2
        'timeout': 8,  # Turunkan dari 10 ke 8
        'alert_cooldown': 120,  # Naikkan dari 30 ke 120 detik
        'services': {
            'discord': {
                'enabled': True,
                'webhook_url': os.getenv('DISCORD_WEBHOOK_URL', ''),
                'username': os.getenv('DISCORD_WEBHOOK_USERNAME', 'Security System'),
                'avatar_url': os.getenv('DISCORD_AVATAR_URL', ''),
                'mention_role_id': None,
                'color': 0xff0000,
                'send_image': True,
                'send_video': False,  # DEFAULT: Tidak mengirim video
                'max_images_per_hour': 15,  # Maksimal 15 gambar per jam
                'max_videos_per_hour': 5,   # Maksimal 5 video per jam
            },
            'slack': {
                'enabled': False,
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
                'username': 'Security System',
                'channel': '#security',
                'send_image': True,
                'send_video': False,
            },
            'teams': {
                'enabled': False,
                'webhook_url': os.getenv('TEAMS_WEBHOOK_URL', ''),
                'send_image': True,
                'send_video': False,
            },
            'custom': {
                'enabled': False,
                'webhook_url': os.getenv('CUSTOM_WEBHOOK_URL', ''),
                'token': os.getenv('CUSTOM_WEBHOOK_TOKEN', ''),
                'send_image': True,
                'send_video': False,
            }
        }
    }
}

# ==================== DISPLAY CONFIGURATION ====================
DISPLAY_CONFIG = {
    'window_title': 'Professional CCTV Security System',
    'window_size': (1280, 720),
    'fullscreen': False,
    'always_on_top': False,
    'show_fps': True,
    'show_detection_info': True,
    'show_timestamp': True,
    'grid_color': (50, 50, 50),
    'text_color': (255, 255, 255),
    'alert_color': (0, 0, 255),
    'font_scale': 0.6,
    'font_thickness': 2,

    'layouts': {
        '2x2': {'rows': 2, 'cols': 2, 'cell_width': 640, 'cell_height': 360},
        '1x4': {'rows': 1, 'cols': 4, 'cell_width': 320, 'cell_height': 720},
        '4x1': {'rows': 4, 'cols': 1, 'cell_width': 1280, 'cell_height': 180},
        '1x1': {'rows': 1, 'cols': 1, 'cell_width': 1280, 'cell_height': 720}
    }
}

# ==================== RECORDING CONFIGURATION ====================
RECORDING_CONFIG = {
    'enabled': True,
    'format': 'mp4',
    'codec': 'mp4v',
    'fps': 15,  # Turunkan dari 20 ke 15
    'quality': 85,  # Turunkan dari 95 ke 85
    'max_file_size_mb': 100,  # Turunkan dari 200 ke 100
    'auto_cleanup': True,
    'cleanup_days': 7,  # Turunkan dari 30 ke 7 hari
    'max_storage_gb': 5,  # Turunkan dari 10 ke 5 GB
    'compression_level': 6,  # Naikkan kompresi dari 3 ke 6
}

# ==================== VISUAL CONFIGURATION ====================
DETECTION_COLORS = {
    'person': (0, 255, 0),
    'car': (255, 0, 0),
    'motorcycle': (0, 0, 255),
    'bicycle': (255, 255, 0),
    'bus': (255, 0, 255),
    'truck': (0, 255, 255),
    'cat': (128, 255, 128),
    'dog': (128, 255, 128)
}

ZONE_COLORS = {
    'entry': (0, 0, 255),
    'perimeter': (255, 165, 0),
    'outdoor': (0, 255, 255),
    'indoor': (0, 255, 0)
}

# ==================== TIMEZONE UTILITIES ====================
def get_wib_time():
    """Get current time in WIB timezone"""
    from datetime import datetime
    return datetime.now(TIMEZONE)

def get_wib_timestamp(format_str="%Y-%m-%d %H:%M:%S"):
    """Get formatted WIB timestamp"""
    return get_wib_time().strftime(format_str)

def get_wib_filename_timestamp():
    """Get WIB timestamp for filenames"""
    return get_wib_time().strftime("%Y%m%d_%H%M%S")

# ==================== UTILITY FUNCTIONS ====================
def get_active_performance_config():
    """Get active performance configuration based on hardware"""
    try:
        import torch
        if torch.cuda.is_available() and HARDWARE_CONFIG.get('use_gpu', True):
            return PERFORMANCE_CONFIG['gpu_profile']
    except ImportError:
        pass
    return PERFORMANCE_CONFIG['cpu_profile']

def get_yolo_config_for_hardware():
    """Get YOLO configuration optimized for current hardware"""
    config = YOLO_CONFIG.copy()
    perf_config = get_active_performance_config()

    # Update detection interval based on performance
    config['detection_interval'] = max(1, perf_config['frame_skip_detection'] // 2)

    return config

def get_hardware_info():
    """Get current hardware configuration summary"""
    return {
        'use_gpu': HARDWARE_CONFIG['use_gpu'],
        'force_cpu': HARDWARE_CONFIG['force_cpu'],
        'gpu_device': HARDWARE_CONFIG['gpu_device'],
        'performance_profile': 'gpu' if HARDWARE_CONFIG['use_gpu'] else 'cpu',
        'cuda_optimizations': HARDWARE_CONFIG['cuda_optimizations'] if HARDWARE_CONFIG['use_gpu'] else None
    }

def get_display_layout(layout_name):
    """Get display layout configuration"""
    return DISPLAY_CONFIG['layouts'].get(layout_name, DISPLAY_CONFIG['layouts']['2x2'])

def validate_config():
    """Validate configuration settings"""
    errors = []

    # Validate environment variables
    required_env_vars = ['CAMERA_IP', 'CAMERA_USERNAME', 'CAMERA_PASSWORD']
    for var in required_env_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")

    # Validate alert services
    if ALERT_CONFIG['telegram_bot']['enabled']:
        if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('TELEGRAM_CHAT_ID'):
            errors.append("Telegram enabled but missing BOT_TOKEN or CHAT_ID")

    if ALERT_CONFIG['webhook_alerts']['enabled']:
        discord_config = ALERT_CONFIG['webhook_alerts']['services']['discord']
        if discord_config['enabled'] and not os.getenv('DISCORD_WEBHOOK_URL'):
            errors.append("Discord webhook enabled but missing WEBHOOK_URL")

    return errors

# ==================== CONFIGURATION SUMMARY ====================
CONFIG_SUMMARY = {
    'total_cameras': len(CAMERAS_CONFIG),
    'gpu_enabled': HARDWARE_CONFIG['use_gpu'],
    'detection_enabled': YOLO_CONFIG['detection_enabled'],
    'security_armed': SECURITY_CONFIG['armed'],
    'alert_methods': sum([
        ALERT_CONFIG['email_alerts']['enabled'],
        ALERT_CONFIG['sound_alerts']['enabled'],
        ALERT_CONFIG['telegram_bot']['enabled']
    ])
}

# ==================== HELPER FUNCTIONS ====================
def get_alert_classes_for_zone(zone: str) -> list:
    """Get alert classes for specific security zone"""
    zone_classes = YOLO_CONFIG.get('zone_alert_classes', {})
    return zone_classes.get(zone, YOLO_CONFIG['alert_classes'])

def should_send_alert(detection_type: str, zone: str) -> bool:
    """Check if detection should trigger alert based on zone and type"""
    allowed_classes = get_alert_classes_for_zone(zone)
    return detection_type in allowed_classes
