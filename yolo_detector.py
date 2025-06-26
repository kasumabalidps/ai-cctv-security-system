import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import List, Dict, Optional, Tuple
import os

# Import YOLO with error handling
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    TORCH_AVAILABLE = False
    YOLO = None
    torch = None

from config import (
    YOLO_CONFIG, DETECTION_COLORS, HARDWARE_CONFIG,
    get_active_performance_config, get_yolo_config_for_hardware
)

logger = logging.getLogger(__name__)

class OptimizedYOLODetector:
    def __init__(self):
        self.model = None
        self.detection_enabled = YOLO_CONFIG['detection_enabled']
        self.confidence_threshold = YOLO_CONFIG['confidence_threshold']
        self.detection_classes = YOLO_CONFIG['detection_classes']
        self.alert_classes = YOLO_CONFIG['alert_classes']
        self.show_labels = YOLO_CONFIG['show_labels']
        self.show_confidence = YOLO_CONFIG['show_confidence']

        # Hardware configuration
        self.hardware_config = HARDWARE_CONFIG
        self.perf_config = get_active_performance_config()
        self.yolo_hw_config = get_yolo_config_for_hardware()
        self.use_gpu = False
        self.device = 'cpu'
        self.gpu_info = None

        # Performance optimizations
        self.detection_interval = YOLO_CONFIG.get('detection_interval', 3)
        self.frame_counter = {}
        self.async_detection = YOLO_CONFIG.get('async_detection', True)
        self.max_detections = YOLO_CONFIG.get('max_detections', 10)

        # Async processing
        self.detection_queue = queue.Queue(maxsize=10)
        self.result_cache = {}
        self.cache_timeout = 0.5

        # Anti-spam system for static objects
        anti_spam_config = YOLO_CONFIG.get('anti_spam', {})
        self.anti_spam_enabled = anti_spam_config.get('enabled', True)
        self.object_tracker = {}  # Track object positions per camera
        self.alert_cooldown = {}  # Cooldown per object type per camera
        self.movement_threshold = anti_spam_config.get('movement_threshold', 50)  # Pixels movement to trigger new alert
        self.static_object_timeout = anti_spam_config.get('static_object_timeout', 300)  # 5 minutes before re-alerting static objects
        self.alert_spam_cooldown = anti_spam_config.get('alert_spam_cooldown', 30)  # 30 seconds cooldown for same object type
        self.cleanup_interval = anti_spam_config.get('cleanup_interval', 600)  # 10 minutes cleanup

        # Threading for async detection
        self.detection_thread = None
        self.running = False

        # Load model and configure hardware
        self.setup_hardware()
        self.load_model()

        # Start async detection thread if enabled
        if self.async_detection and self.is_model_loaded():
            self.start_async_detection()

    def setup_hardware(self):
        """Setup hardware configuration (GPU/CPU)"""
        logger.info("🔧 Configuring hardware for YOLO detection...")

        if self.hardware_config.get('force_cpu', False):
            self.use_gpu = False
            self.device = 'cpu'
            logger.info("🖥️ Forced CPU mode - GPU disabled by config")
            return

        if self.hardware_config.get('use_gpu', False) and TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_device = self.hardware_config.get('gpu_device', 0)

                    if gpu_device < torch.cuda.device_count():
                        self.use_gpu = True
                        self.device = f'cuda:{gpu_device}'

                        self.gpu_info = {
                            'name': torch.cuda.get_device_name(gpu_device),
                            'memory_total': torch.cuda.get_device_properties(gpu_device).total_memory / 1024**3,
                            'device_index': gpu_device,
                            'compute_capability': torch.cuda.get_device_properties(gpu_device).major
                        }

                        logger.info(f"🚀 GPU Mode Enabled!")
                        logger.info(f"   GPU: {self.gpu_info['name']}")
                        logger.info(f"   Memory: {self.gpu_info['memory_total']:.1f} GB")
                        logger.info(f"   Device: cuda:{gpu_device}")

                        self.setup_cuda_optimizations()

                    else:
                        logger.warning(f"⚠️ GPU device {gpu_device} not available, using CPU")
                        self.use_gpu = False
                        self.device = 'cpu'
                else:
                    logger.warning("⚠️ CUDA not available, using CPU")
                    self.use_gpu = False
                    self.device = 'cpu'

            except Exception as e:
                logger.error(f"❌ GPU setup failed: {e}")
                self.use_gpu = False
                self.device = 'cpu'
        else:
            self.use_gpu = False
            self.device = 'cpu'
            logger.info("🖥️ CPU Mode - GPU disabled or not requested")

        if self.use_gpu:
            logger.info(f"✅ Hardware: GPU ({self.device})")
        else:
            logger.info(f"✅ Hardware: CPU")

    def setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations"""
        if not self.use_gpu or not TORCH_AVAILABLE:
            return

        try:
            cuda_config = self.hardware_config.get('cuda_optimizations', {})

            memory_fraction = cuda_config.get('memory_fraction', 0.8)
            if memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"🔧 GPU memory fraction: {memory_fraction}")

            if cuda_config.get('allow_growth', True):
                pass

            torch.backends.cudnn.benchmark = True
            logger.info("🔧 CUDA optimizations applied")

        except Exception as e:
            logger.warning(f"⚠️ CUDA optimization failed: {e}")

    def load_model(self):
        """Load YOLO model with hardware-specific optimizations"""
        if not YOLO_AVAILABLE:
            logger.warning("❌ Ultralytics not installed. YOLO detection disabled.")
            self.detection_enabled = False
            return False

        try:
            model_path = YOLO_CONFIG['model_path']
            logger.info(f"📦 Loading YOLO model: {model_path}")

            self.model = YOLO(model_path)

            if self.use_gpu:
                self.model.to(self.device)
                logger.info(f"🚀 Model loaded on GPU: {self.device}")
                self.apply_gpu_optimizations()
            else:
                self.apply_cpu_optimizations()
                logger.info("🖥️ Model loaded on CPU")

            self.warmup_model()

            logger.info("✅ YOLO model ready for detection")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {str(e)}")
            self.detection_enabled = False
            return False

    def apply_gpu_optimizations(self):
        """Apply GPU-specific optimizations"""
        if not self.use_gpu or not hasattr(self.model, 'model'):
            return

        try:
            cuda_config = self.hardware_config.get('cuda_optimizations', {})

            self.model.model.eval()

            if cuda_config.get('use_half_precision', True):
                try:
                    self.model.model.half()
                    logger.info("🔧 Half precision (FP16) enabled")
                except Exception as e:
                    logger.warning(f"⚠️ Half precision not supported: {e}")

            if cuda_config.get('use_tensorrt', False):
                try:
                    logger.info("🔧 TensorRT optimization requested (requires additional setup)")
                except Exception as e:
                    logger.warning(f"⚠️ TensorRT optimization failed: {e}")

        except Exception as e:
            logger.warning(f"⚠️ GPU optimization failed: {e}")

    def apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations"""
        try:
            cpu_config = self.hardware_config.get('cpu_optimizations', {})

            num_threads = cpu_config.get('num_threads', 4)
            if TORCH_AVAILABLE:
                torch.set_num_threads(num_threads)
                logger.info(f"🔧 CPU threads: {num_threads}")

            if cpu_config.get('use_mkldnn', True) and TORCH_AVAILABLE:
                try:
                    torch.backends.mkldnn.enabled = True
                    logger.info("🔧 Intel MKL-DNN enabled")
                except:
                    pass

        except Exception as e:
            logger.warning(f"⚠️ CPU optimization failed: {e}")

    def warmup_model(self):
        """Warmup model for consistent performance"""
        if not self.is_model_loaded():
            return

        try:
            warmup_iterations = self.yolo_hw_config.get('warmup_iterations', 5)
            input_size = self.yolo_hw_config.get('input_size', 640)

            dummy_frame = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

            logger.info(f"🔥 Warming up model ({warmup_iterations} iterations)...")

            for i in range(warmup_iterations):
                _ = self.model(dummy_frame, conf=0.5, verbose=False)

            logger.info("✅ Model warmup completed")

        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")

    def start_async_detection(self):
        """Start async detection thread"""
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.running = True
            self.detection_thread = threading.Thread(target=self._async_detection_worker, daemon=True)
            self.detection_thread.start()
            logger.info("🔄 Async YOLO detection thread started")

    def stop_async_detection(self):
        """Stop async detection thread"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
            logger.info("⏹️ Async YOLO detection thread stopped")

    def _async_detection_worker(self):
        """Async detection worker thread"""
        while self.running:
            try:
                if not self.detection_queue.empty():
                    frame, camera_id, timestamp, confidence_threshold = self.detection_queue.get(timeout=0.1)

                    detections = self._run_detection(frame, confidence_threshold)

                    self.result_cache[camera_id] = {
                        'detections': detections,
                        'timestamp': timestamp
                    }
                else:
                    time.sleep(0.01)

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Async detection error: {e}")
                time.sleep(0.1)

    def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for camera_id, cache_data in self.result_cache.items():
            if current_time - cache_data['timestamp'] > self.cache_timeout:
                expired_keys.append(camera_id)

        for key in expired_keys:
            del self.result_cache[key]

    def detect_objects(self, frame: np.ndarray, camera_name: str, camera_id: str = None, night_mode: bool = False) -> Tuple[np.ndarray, List]:
        """Detect objects with hardware-specific optimizations and night mode support"""
        if not self.is_model_loaded() or not self.detection_enabled:
            return frame, []

        if camera_id not in self.frame_counter:
            self.frame_counter[camera_id] = 0

        self.frame_counter[camera_id] += 1

        should_detect = (self.frame_counter[camera_id] % self.detection_interval) == 0

        detections = []
        alerts = []

        if should_detect:
            # Dynamic confidence threshold based on night mode
            if night_mode:
                confidence_threshold = YOLO_CONFIG.get('confidence_threshold_night', 0.65)
            else:
                # Use day threshold for better detection
                confidence_threshold = YOLO_CONFIG.get('confidence_threshold', 0.55)

            if self.async_detection:
                detections = self._get_async_detection(frame, camera_id, confidence_threshold)
            else:
                detections = self._run_detection(frame, confidence_threshold)
        else:
            if camera_id in self.result_cache:
                cache_data = self.result_cache[camera_id]
                if time.time() - cache_data['timestamp'] < self.cache_timeout:
                    detections = cache_data['detections']

        processed_frame = self._draw_detections(frame.copy(), detections, camera_name, night_mode)
        alerts = self._generate_alerts(detections, camera_id)

        return processed_frame, alerts

    def _get_async_detection(self, frame: np.ndarray, camera_id: str, confidence_threshold: float) -> List:
        """Get detection using async processing"""
        current_time = time.time()

        if not self.detection_queue.full():
            try:
                input_size = self.yolo_hw_config.get('input_size', 640)
                small_frame = cv2.resize(frame, (input_size, input_size))
                self.detection_queue.put((small_frame, camera_id, current_time, confidence_threshold), block=False)
            except queue.Full:
                pass

        if camera_id in self.result_cache:
            cache_data = self.result_cache[camera_id]
            if current_time - cache_data['timestamp'] < self.cache_timeout:
                return cache_data['detections']

        return []

    def _run_detection(self, frame: np.ndarray, confidence_threshold: float) -> List:
        """Run YOLO detection on frame with hardware optimizations"""
        try:
            height, width = frame.shape[:2]

            target_size = self.yolo_hw_config.get('input_size', 640)
            batch_size = self.yolo_hw_config.get('batch_size', 1)

            if max(width, height) > target_size:
                scale = target_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                detection_frame = cv2.resize(frame, (new_width, new_height))
                scale_x = width / new_width
                scale_y = height / new_height
            else:
                detection_frame = frame
                scale_x = scale_y = 1.0

            results = self.model(
                detection_frame,
                conf=confidence_threshold,
                iou=YOLO_CONFIG.get('nms_threshold', 0.45),
                max_det=self.max_detections,
                verbose=False,
                device=self.device
            )

            detections = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                        if i >= self.max_detections:
                            break

                        class_name = self.model.names[int(cls)]

                        if class_name in self.detection_classes:
                            x1, y1, x2, y2 = box
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)

                            detections.append({
                                'class': class_name,
                                'confidence': float(score),
                                'bbox': (x1, y1, x2, y2)
                            })

            return detections

        except Exception as e:
            logger.error(f"❌ Detection error: {str(e)}")
            return []

    def _draw_detections(self, frame: np.ndarray, detections: List, camera_name: str, night_mode: bool) -> np.ndarray:
        """Draw clean professional detection boxes and labels with night mode support"""
        if not detections:
            return frame

        try:
            for detection in detections:
                class_name = detection['class']
                confidence = detection['confidence']
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox

                # Enhanced colors for night mode (brighter, more visible)
                if night_mode:
                    base_colors = {
                        'person': (0, 255, 0),      # Bright green for night
                        'car': (255, 150, 0),       # Bright orange for night
                        'motorcycle': (0, 150, 255), # Bright blue for night
                        'bicycle': (255, 255, 0),    # Bright yellow for night
                        'bus': (255, 0, 255),       # Bright magenta for night
                        'truck': (0, 255, 255),     # Bright cyan for night
                        'cat': (150, 255, 150),     # Bright light green for night
                        'dog': (150, 255, 150)      # Bright light green for night
                    }
                else:
                    # Subtle colors for day mode
                    base_colors = {
                        'person': (0, 180, 0),      # Subtle green
                        'car': (180, 100, 0),       # Subtle blue
                        'motorcycle': (0, 100, 180), # Subtle red
                        'bicycle': (180, 180, 0),    # Subtle cyan
                        'bus': (180, 0, 180),       # Subtle magenta
                        'truck': (0, 180, 180),     # Subtle yellow
                        'cat': (100, 180, 100),     # Light green
                        'dog': (100, 180, 100)      # Light green
                    }

                color = base_colors.get(class_name, (150, 150, 150))

                # Thicker lines for night mode for better visibility
                thickness = 3 if night_mode else (2 if confidence > 0.7 else 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Enhanced labels for night mode
                if self.show_labels:
                    if self.show_confidence:
                        label = f"{class_name} {confidence:.1f}"
                        if night_mode:
                            label += " 🌙"  # Night mode indicator
                    else:
                        label = class_name

                    # Measure text for clean background
                    font_scale = 0.6 if night_mode else 0.5
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )

                    # Enhanced background for night mode
                    bg_alpha = 0.8 if night_mode else 0.7
                    label_bg = frame[y1-label_height-8:y1, x1:x1+label_width+8].copy()
                    overlay = np.zeros_like(label_bg)
                    overlay[:] = color
                    cv2.addWeighted(label_bg, 1-bg_alpha, overlay, bg_alpha, 0, label_bg)
                    frame[y1-label_height-8:y1, x1:x1+label_width+8] = label_bg

                    # Brighter text for night mode
                    text_color = (255, 255, 255) if night_mode else (255, 255, 255)
                    text_thickness = 2 if night_mode else 1
                    cv2.putText(frame, label, (x1+4, y1-4),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            return frame

        except Exception as e:
            logger.error(f"❌ Drawing error: {str(e)}")
            return frame

    def _generate_alerts(self, detections: List, camera_id: str = None) -> List:
        """Generate smart alerts that prevent spam for static objects"""
        alerts = []
        current_time = time.time()

        # If anti-spam is disabled, use simple alert generation
        if not self.anti_spam_enabled:
            for detection in detections:
                if detection['class'] in self.alert_classes:
                    alerts.append({
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'timestamp': current_time
                    })
            return alerts

        # Initialize tracking for this camera if needed
        if camera_id not in self.object_tracker:
            self.object_tracker[camera_id] = {}
            self.alert_cooldown[camera_id] = {}

        for detection in detections:
            if detection['class'] not in self.alert_classes:
                continue

            class_name = detection['class']
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if this object type is in cooldown
            cooldown_key = f"{camera_id}_{class_name}"
            if cooldown_key in self.alert_cooldown[camera_id]:
                last_alert_time = self.alert_cooldown[camera_id][cooldown_key]
                if current_time - last_alert_time < self.alert_spam_cooldown:
                    continue  # Skip alert, still in cooldown

            # Check for movement vs static objects
            should_alert = False
            object_key = f"{class_name}_{center_x//100}_{center_y//100}"  # Grid-based tracking

            if object_key in self.object_tracker[camera_id]:
                # Object exists, check if it moved significantly
                last_pos = self.object_tracker[camera_id][object_key]
                last_x, last_y, last_time = last_pos['x'], last_pos['y'], last_pos['time']

                distance_moved = ((center_x - last_x) ** 2 + (center_y - last_y) ** 2) ** 0.5
                time_since_last = current_time - last_time

                # Alert conditions:
                # 1. Object moved significantly (new position)
                # 2. Static object timeout reached (re-alert after 5 minutes)
                if distance_moved > self.movement_threshold:
                    should_alert = True
                    logger.info(f"🚶 {class_name} moved {distance_moved:.0f}px on camera {camera_id}")
                elif time_since_last > self.static_object_timeout:
                    should_alert = True
                    logger.info(f"⏰ {class_name} static timeout reached on camera {camera_id}")

            else:
                # New object detected
                should_alert = True
                logger.info(f"🆕 New {class_name} detected on camera {camera_id}")

            # Update object tracking
            self.object_tracker[camera_id][object_key] = {
                'x': center_x,
                'y': center_y,
                'time': current_time,
                'class': class_name
            }

            # Generate alert if conditions met
            if should_alert:
                alerts.append({
                    'class': class_name,
                    'confidence': detection['confidence'],
                    'timestamp': current_time,
                    'position': (center_x, center_y),
                    'movement_type': 'new' if object_key not in self.object_tracker[camera_id] else 'moved'
                })

                # Set cooldown for this object type
                self.alert_cooldown[camera_id][cooldown_key] = current_time

        # Clean up old tracking data (older than 10 minutes)
        self._cleanup_old_tracking(camera_id, current_time)

        return alerts

    def _cleanup_old_tracking(self, camera_id: str, current_time: float):
        """Clean up old object tracking data"""
        if camera_id not in self.object_tracker:
            return

        cleanup_timeout = self.cleanup_interval
        keys_to_remove = []

        for object_key, data in self.object_tracker[camera_id].items():
            if current_time - data['time'] > cleanup_timeout:
                keys_to_remove.append(object_key)

        for key in keys_to_remove:
            del self.object_tracker[camera_id][key]

        # Also cleanup cooldown data
        cooldown_keys_to_remove = []
        for cooldown_key, last_time in self.alert_cooldown[camera_id].items():
            if current_time - last_time > cleanup_timeout:
                cooldown_keys_to_remove.append(cooldown_key)

        for key in cooldown_keys_to_remove:
            del self.alert_cooldown[camera_id][key]

    def toggle_detection(self):
        """Toggle detection on/off"""
        self.detection_enabled = not self.detection_enabled
        logger.info(f"🔄 YOLO detection {'enabled' if self.detection_enabled else 'disabled'}")

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and YOLO_AVAILABLE

    def get_detection_stats(self) -> Dict:
        """Get detection statistics including anti-spam info"""
        stats = {
            'model_loaded': self.is_model_loaded(),
            'detection_enabled': self.detection_enabled,
            'hardware': 'GPU' if self.use_gpu else 'CPU',
            'device': self.device,
            'async_enabled': self.async_detection,
            'confidence_threshold': self.confidence_threshold,
            'detection_classes': len(self.detection_classes),
            'alert_classes': len(self.alert_classes),
            'performance_profile': 'GPU' if self.use_gpu else 'CPU'
        }

        # Add anti-spam statistics
        total_tracked_objects = sum(len(camera_objects) for camera_objects in self.object_tracker.values())
        total_cooldowns = sum(len(camera_cooldowns) for camera_cooldowns in self.alert_cooldown.values())

        stats.update({
            'anti_spam': {
                'tracked_objects': total_tracked_objects,
                'active_cooldowns': total_cooldowns,
                'movement_threshold': self.movement_threshold,
                'static_timeout': self.static_object_timeout,
                'spam_cooldown': self.alert_spam_cooldown
            }
        })

        return stats

    def get_hardware_info(self) -> Dict:
        """Get detailed hardware information"""
        info = {
            'use_gpu': self.use_gpu,
            'device': self.device,
            'torch_available': TORCH_AVAILABLE,
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
        }

        if self.use_gpu and self.gpu_info:
            info.update(self.gpu_info)

        return info

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"🎯 Confidence threshold set to: {self.confidence_threshold}")

    def set_detection_interval(self, interval: int):
        """Set detection interval (frames to skip)"""
        self.detection_interval = max(1, interval)
        logger.info(f"⏱️ Detection interval set to: {self.detection_interval}")

    def switch_device(self, use_gpu: bool):
        """Switch between GPU and CPU at runtime"""
        if use_gpu == self.use_gpu:
            return

        logger.info(f"🔄 Switching to {'GPU' if use_gpu else 'CPU'} mode...")

        self.hardware_config['use_gpu'] = use_gpu

        self.setup_hardware()
        if self.is_model_loaded():
            self.load_model()
            self.yolo_hw_config = get_yolo_config_for_hardware()
            logger.info(f"✅ Successfully switched to {'GPU' if self.use_gpu else 'CPU'}")
        else:
            logger.error("❌ Failed to switch device")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_async_detection()

# Backward compatibility alias
YOLODetector = OptimizedYOLODetector
