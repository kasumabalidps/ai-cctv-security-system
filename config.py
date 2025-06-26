# Professional Home Security System Configuration
import pytz

# ==================== TIMEZONE CONFIGURATION ====================
TIMEZONE = pytz.timezone('Asia/Jakarta')  # WIB (Western Indonesia Time)

# ==================== CAMERA CONFIGURATION ====================
CAMERAS_CONFIG = [
    {
        'id': '1',
        'ip': '192.168.1.108',
        'username': 'admin',
        'password': 'admin123',
        'channel': 1,
        'name': 'Halaman Kost',
        'security_zone': 'entry',
        'priority': 'high',
        'record_alerts': True
    },
    {
        'id': '2',
        'ip': '192.168.1.108',
        'username': 'admin',
        'password': 'admin123',
        'channel': 2,
        'name': 'Gerbang Kost',
        'security_zone': 'perimeter',
        'priority': 'high',
        'record_alerts': True
    },
    {
        'id': '3',
        'ip': '192.168.1.108',
        'username': 'admin',
        'password': 'admin123',
        'channel': 3,
        'name': 'Gerbang Rumah',
        'security_zone': 'perimeter',
        'priority': 'critical',
        'record_alerts': True
    },
    {
        'id': '4',
        'ip': '192.168.1.108',
        'username': 'admin',
        'password': 'admin123',
        'channel': 4,
        'name': 'Halaman Rumah',
        'security_zone': 'outdoor',
        'priority': 'critical',
        'record_alerts': True
    }
]

# ==================== HARDWARE CONFIGURATION ====================
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

# ==================== YOLO DETECTION CONFIGURATION ====================
YOLO_CONFIG = {
    'model_path': 'yolov8l.pt',
    'confidence_threshold': 0.55,  # Pagi/Siang: deteksi maksimal, false positive rendah
    'confidence_threshold_night': 0.65,  # Malam (IR): menghindari bayangan/pantulan
    'confidence_threshold_24h': 0.6,  # Aman all day - trade-off terbaik
    'nms_threshold': 0.45,
    'detection_enabled': True,
    'async_detection': False,
    'show_labels': True,
    'show_confidence': True,
    'auto_night_mode': True,

    'detection_classes': ['person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'cat', 'dog'],
    'alert_classes': ['person', 'car', 'motorcycle'],
    'max_detections': 15,
    'detection_interval': 1,

    'anti_spam': {
        'enabled': True,
        'movement_threshold': 50,
        'static_object_timeout': 300,
        'alert_spam_cooldown': 30,
        'cleanup_interval': 600
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

# ==================== PERFORMANCE CONFIGURATION ====================
PERFORMANCE_CONFIG = {
    'gpu_profile': {
        'target_fps': 20,
        'detection_fps': 10,
        'frame_skip_detection': 2,
        'buffer_size': 1,
        'thread_pool_size': 6,
        'memory_optimization': True,
        'prefetch_frames': True,
    },

    'cpu_profile': {
        'target_fps': 15,
        'detection_fps': 5,
        'frame_skip_detection': 3,
        'buffer_size': 1,
        'thread_pool_size': 4,
        'memory_optimization': True,
        'prefetch_frames': False,
    }
}

# ==================== SECURITY CONFIGURATION ====================
SECURITY_CONFIG = {
    'armed': True,
    'sensitivity': 'high',
    'alert_cooldown': 30,
    'recording_duration': 30,
    'backup_alerts': True,
    'continuous_recording': False,
    'motion_sensitivity': 0.4,
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
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'email': 'your-email@gmail.com',
        'password': 'your-app-password',
        'recipients': ['security@yourdomain.com']
    },

    'sound_alerts': {
        'enabled': True,
        'alert_sound': 'assets/alert.wav',
        'volume': 0.8
    },

    'telegram_bot': {
        'enabled': True,
        'bot_token': '8199303892:AAG9gNITebF7Uid9-khQgNK6uWu6NN42TRY',
        'chat_id': '6398781481',
        'send_photo': True,
        'send_video': True,
        'video_max_size_mb': 48,  # Telegram limit is 50MB, leave margin
        'video_timeout': 60,      # Timeout for video upload
        'enhanced_formatting': True
    },

    'webhook_alerts': {
        'enabled': False,
        'services': {
            'discord': {
                'enabled': False,
                'webhook_url': 'https://discord.com/api/webhooks/1387816647750062202/xXmS7o-JDFd81Zvg78e3s_9aKBIDRcGuCRMEkg_Ywu6Ai0Vje3NBNIiN3xu_vnu4C5b5',
                'username': 'Security System',
                'avatar_url': 'https://cdn-icons-png.flaticon.com/512/2913/2913095.png',
                'mention_role_id': None,  # Optional: Discord role ID to mention
                'color': 0xff0000,  # Red color for alerts
                'send_image': True  # Send screenshot with alert
            },
            'slack': {
                'enabled': False,
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                'channel': '#security-alerts',
                'username': 'Security Bot',
                'icon_emoji': ':warning:',
                'send_image': True
            },
            'teams': {
                'enabled': False,
                'webhook_url': 'https://outlook.office.com/webhook/YOUR_TEAMS_WEBHOOK',
                'title': 'Security Alert',
                'theme_color': 'FF0000',
                'send_image': False  # Teams has limited image support
            },
            'custom': {
                'enabled': False,
                'webhook_url': 'https://your-custom-endpoint.com/webhook',
                'method': 'POST',
                'headers': {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer YOUR_TOKEN'
                },
                'send_image': True
            }
        },
        'retry_attempts': 3,
        'timeout': 10,
        'include_screenshot': True,
        'alert_cooldown': 30  # Seconds between webhook alerts for same detection
    },

    'push_notifications': {
        'enabled': False,
        'service': 'firebase'
    }
}

# ==================== DISPLAY CONFIGURATION ====================
DISPLAY_CONFIG = {
    'window_width': 1280,
    'window_height': 720,
    'full_hd_screenshots': True,
    'compression_quality': 95,

    'grid_layouts': {
        '2x2': (640, 360),
        '1x4': (320, 180),
        '4x1': (320, 180),
        '1x1': (1280, 720)
    },

    'directories': {
        'screenshots': 'screenshots',
        'alerts': 'alerts',
        'recordings': 'recordings',
        'logs': 'logs'
    },

    # Legacy support
    'grid_size_2x2': (640, 360),
    'grid_size_1x4': (320, 180),
    'grid_size_4x1': (320, 180),
    'grid_size_1x1': (1280, 720),
    'screenshot_dir': 'screenshots',
    'alerts_dir': 'alerts',
    'recordings_dir': 'recordings',
    'logs_dir': 'logs'
}

# ==================== RECORDING CONFIGURATION ====================
RECORDING_CONFIG = {
    'enabled': True,
    'format': 'mp4',
    'codec': 'h264',
    'fps': 15,
    'resolution': (1280, 720),
    'max_file_size_mb': 100,
    'auto_cleanup_days': 30,
    'backup_to_cloud': False
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
    """Get current time in WIB (Asia/Jakarta) timezone"""
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
    """Get performance configuration based on hardware settings"""
    if HARDWARE_CONFIG['use_gpu'] and not HARDWARE_CONFIG['force_cpu']:
        return PERFORMANCE_CONFIG['gpu_profile']
    else:
        return PERFORMANCE_CONFIG['cpu_profile']

def get_yolo_config_for_hardware():
    """Get YOLO configuration based on hardware"""
    if HARDWARE_CONFIG['use_gpu'] and not HARDWARE_CONFIG['force_cpu']:
        return YOLO_CONFIG['gpu_settings']
    else:
        return YOLO_CONFIG['cpu_settings']

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
    layouts = {
        "2x2": {"grid": (2, 2), "size": DISPLAY_CONFIG['grid_layouts']['2x2']},
        "1x4": {"grid": (1, 4), "size": DISPLAY_CONFIG['grid_layouts']['1x4']},
        "4x1": {"grid": (4, 1), "size": DISPLAY_CONFIG['grid_layouts']['4x1']},
        "1x1": {"grid": (1, 1), "size": DISPLAY_CONFIG['grid_layouts']['1x1']}
    }
    return layouts.get(layout_name, layouts["2x2"])

def validate_config():
    """Validate configuration settings"""
    errors = []

    if not CAMERAS_CONFIG:
        errors.append("No cameras configured")

    if HARDWARE_CONFIG['gpu_device'] < 0:
        errors.append("Invalid GPU device index")

    if YOLO_CONFIG['confidence_threshold'] < 0 or YOLO_CONFIG['confidence_threshold'] > 1:
        errors.append("Invalid confidence threshold")

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
