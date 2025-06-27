import cv2
import threading
import time
import numpy as np
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime

from config import CAMERAS_CONFIG, YOLO_CONFIG, SECURITY_CONFIG, HARDWARE_CONFIG, get_active_performance_config, get_wib_timestamp, get_wib_filename_timestamp
from yolo_detector import OptimizedYOLODetector
from security_manager import SecurityManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalSecuritySystem:
    def __init__(self):
        self.cameras = {}
        self.running = False
        self.display_thread = None
        self.current_layout = "2x2"  # Only "2x2" and "1x1" for home security
        self.fullscreen_camera = None
        self.fps_counter = {}
        self.screenshot_dir = 'screenshots'

        # Performance tracking
        self.frame_times = {}
        self.detection_times = {}
        self.last_fps_update = time.time()

        self.yolo_detector = OptimizedYOLODetector()
        self.security_manager = SecurityManager()

        self.detection_enabled = True
        self.yolo_detector.detection_enabled = True

        if SECURITY_CONFIG.get('force_armed_on_startup', True):
            self.security_manager.armed = True

        if SECURITY_CONFIG.get('auto_enable_detection', True):
            if self.yolo_detector.is_model_loaded():
                self.yolo_detector.detection_enabled = True
                self.detection_enabled = True

        self.hardware_config = HARDWARE_CONFIG
        self.performance_config = get_active_performance_config()

        self.show_stats = False
        self.show_help = False
        self.night_mode = SECURITY_CONFIG.get('night_mode', False)

        for directory in ['screenshots', 'alerts', 'recordings', 'logs']:
            os.makedirs(directory, exist_ok=True)

        self.log_hardware_info()

        logger.info("🔒 HOME SECURITY MODE - Detection and Security ALWAYS ON")
        logger.info(f"🔒 Security Armed: {self.security_manager.is_armed()}")
        logger.info(f"🤖 Detection Active: {self.detection_enabled and self.yolo_detector.detection_enabled}")
        logger.info("Professional Security System initialized")

    def log_hardware_info(self):
        hardware_info = self.yolo_detector.get_hardware_info()

        logger.info("🔧 Hardware Configuration:")
        logger.info(f"   Mode: {'GPU' if hardware_info.get('use_gpu') else 'CPU'}")
        logger.info(f"   Device: {hardware_info.get('device', 'cpu')}")

        if hardware_info.get('use_gpu') and hardware_info.get('name'):
            logger.info(f"   GPU: {hardware_info['name']}")
            logger.info(f"   Memory: {hardware_info.get('memory_total', 0):.1f} GB")

        perf_profile = 'GPU' if self.hardware_config.get('use_gpu') else 'CPU'
        logger.info(f"   Performance Profile: {perf_profile}")

    def add_camera(self, camera_id: str, ip: str, username: str, password: str,
                   port: int = 554, channel: int = 1, name: str = None,
                   security_zone: str = "unknown", priority: str = "medium"):
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0"

        self.cameras[camera_id] = {
            'url': rtsp_url,
            'ip': ip,
            'channel': channel,
            'name': name or f"Camera {channel}",
            'security_zone': security_zone,
            'priority': priority,
            'cap': None,
            'frame': None,
            'connected': False,
            'thread': None,
            'last_frame_time': time.time(),
            'frame_count': 0,
            'fps_time': time.time(),
            'detections': [],
            'alerts': [],
            'performance': {
                'avg_fps': 0,
                'detection_fps': 0,
                'connection_time': 0
            },
            'night_mode': self.night_mode
        }

        self.fps_counter[camera_id] = 0
        self.frame_times[camera_id] = []
        self.detection_times[camera_id] = []

        logger.info(f"Added camera {camera_id} ({name}) - Zone: {security_zone}, Priority: {priority}")

    def connect_camera(self, camera_id: str) -> bool:
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return False

        camera = self.cameras[camera_id]

        try:
            logger.info(f"Connecting to {camera['name']} at {camera['ip']}")
            start_time = time.time()

            cap = cv2.VideoCapture(camera['url'])

            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.performance_config['buffer_size'])
            cap.set(cv2.CAP_PROP_FPS, self.performance_config['target_fps'])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_BITRATE, 10000000)

            ret, frame = cap.read()
            connection_time = time.time() - start_time

            if ret and frame is not None:
                camera['cap'] = cap
                camera['connected'] = True
                camera['frame'] = frame
                camera['performance']['connection_time'] = connection_time

                logger.info(f"✓ {camera['name']} connected in {connection_time:.2f}s")
                return True
            else:
                cap.release()
                logger.error(f"✗ {camera['name']} connection failed")
                return False

        except Exception as e:
            logger.error(f"Connection error for {camera_id}: {str(e)}")
            return False

    def camera_thread(self, camera_id: str):
        camera = self.cameras[camera_id]
        consecutive_failures = 0
        max_failures = 10
        detection_frame_counter = 0

        while self.running:
            if not camera['connected']:
                if self.connect_camera(camera_id):
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error(f"Camera {camera_id} failed permanently")
                        break
                    time.sleep(2)
                    continue

            try:
                frame_start = time.time()
                ret, frame = camera['cap'].read()

                if ret and frame is not None:
                    detection_frame_counter += 1

                    if detection_frame_counter % 60 == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        brightness = np.mean(gray)
                        is_night_mode = brightness < 40

                        if is_night_mode != camera.get('night_mode', False):
                            camera['night_mode'] = is_night_mode
                            mode_text = "NIGHT (IR)" if is_night_mode else "DAY"
                            confidence_used = YOLO_CONFIG.get('confidence_threshold_night', 0.65) if is_night_mode else YOLO_CONFIG.get('confidence_threshold', 0.55)
                            logger.info(f"🌙 {camera['name']}: {mode_text} mode detected (brightness: {brightness:.1f}, confidence: {confidence_used})")

                    detection_start = time.time()

                    if self.detection_enabled and self.yolo_detector.detection_enabled:
                        frame, alerts = self.yolo_detector.detect_objects(
                            frame, camera['name'], camera_id, camera.get('night_mode', False)
                        )

                        # Process security alerts
                        if alerts:
                            camera_config = {
                                'security_zone': camera['security_zone'],
                                'priority': camera['priority'],
                                'record_alerts': True
                            }

                            alert_triggered = self.security_manager.process_detection(
                                camera_id, camera['name'], alerts, frame, camera_config
                            )

                            if alert_triggered:
                                camera['alerts'] = alerts
                                logger.warning(f"🚨 SECURITY ALERT: {camera['name']} - {len(alerts)} objects detected")
                                for alert in alerts:
                                    movement_info = f" ({alert.get('movement_type', 'new')})"
                                    logger.warning(f"   - {alert['class']} (confidence: {alert['confidence']:.2f}){movement_info}")

                    detection_time = time.time() - detection_start

                    camera['frame'] = frame
                    camera['last_frame_time'] = time.time()
                    consecutive_failures = 0

                    frame_time = time.time() - frame_start
                    self.update_performance_stats(camera_id, frame_time, detection_time)

                    self.security_manager.update_recording(camera_id, frame)

                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning(f"{camera['name']} disconnected")
                        camera['connected'] = False
                        if camera['cap']:
                            camera['cap'].release()
                            camera['cap'] = None

            except Exception as e:
                logger.error(f"Camera thread error {camera_id}: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    camera['connected'] = False
                    if camera['cap']:
                        camera['cap'].release()
                        camera['cap'] = None

            target_sleep = 1.0 / self.performance_config['target_fps']
            time.sleep(max(0.01, target_sleep))

    def update_performance_stats(self, camera_id: str, frame_time: float, detection_time: float):
        max_measurements = 30

        if len(self.frame_times[camera_id]) >= max_measurements:
            self.frame_times[camera_id].pop(0)
        self.frame_times[camera_id].append(frame_time)

        if len(self.detection_times[camera_id]) >= max_measurements:
            self.detection_times[camera_id].pop(0)
        self.detection_times[camera_id].append(detection_time)

        if self.frame_times[camera_id]:
            avg_frame_time = sum(self.frame_times[camera_id]) / len(self.frame_times[camera_id])
            self.cameras[camera_id]['performance']['avg_fps'] = 1.0 / max(avg_frame_time, 0.001)

        if self.detection_times[camera_id]:
            avg_detection_time = sum(self.detection_times[camera_id]) / len(self.detection_times[camera_id])
            self.cameras[camera_id]['performance']['detection_fps'] = 1.0 / max(avg_detection_time, 0.001)

    def take_screenshot(self, camera_id: str = None):
        timestamp = get_wib_filename_timestamp()

        if camera_id and camera_id in self.cameras:
            camera = self.cameras[camera_id]
            if camera['frame'] is not None:
                frame = cv2.resize(camera['frame'], (1920, 1080))
                filename = f"{self.screenshot_dir}/HD_{camera['name']}_{timestamp}.jpg"
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.info(f"HD Screenshot saved: {filename}")
        else:
            display_frame = self.create_display()
            filename = f"{self.screenshot_dir}/FULL_VIEW_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Full view screenshot saved: {filename}")

    def create_display(self):
        """Create enhanced display with professional UI"""
        if self.fullscreen_camera and self.fullscreen_camera in self.cameras:
            return self.create_fullscreen_display()
        else:
            return self.create_grid_display()

    def create_fullscreen_display(self):
        """Create clean professional fullscreen display"""
        camera = self.cameras[self.fullscreen_camera]

        if camera['frame'] is not None and camera['connected']:
            frame = camera['frame'].copy()
            # HD resize dengan interpolation terbaik
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

            # Transparent overlay on top of video (not cutting the video)
            overlay = frame.copy()

            # Create semi-transparent header area
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)

            # Camera name with background
            cv2.putText(overlay, camera['name'], (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Security status - minimal
            status_color = (0, 200, 0) if self.security_manager.is_armed() else (100, 100, 100)
            status_text = "●" if self.security_manager.is_armed() else "○"
            cv2.putText(overlay, status_text, (frame.shape[1] - 50, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

            # Timestamp - subtle (WIB)
            timestamp = get_wib_timestamp("%H:%M")
            cv2.putText(overlay, timestamp, (frame.shape[1] - 120, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

            # Alert indicator only if needed
            if camera['alerts']:
                cv2.circle(overlay, (frame.shape[1] - 80, 30), 8, (0, 0, 255), -1)

            # Safe blend overlay with original frame (transparent effect)
            try:
                if frame.shape == overlay.shape and frame.dtype == overlay.dtype:
                    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                else:
                    result = frame  # Fallback if shapes don't match
            except Exception as e:
                logger.debug(f"Overlay blend failed: {e}")
                result = frame

            return result
        else:
            # Clean disconnected view
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:] = (20, 20, 20)

            cv2.putText(frame, camera['name'], (50, frame.shape[0]//2 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (120, 120, 120), 2)
            cv2.putText(frame, "Reconnecting...", (50, frame.shape[0]//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            return frame

    def create_grid_display(self):
        """Create optimized professional grid display"""
        layout = self.get_layout_config()
        grid_rows, grid_cols = layout["grid"]
        cell_size = layout["size"]

        # Pre-calculate dimensions
        total_width = cell_size[0] * grid_cols
        total_height = cell_size[1] * grid_rows
        grid = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # Cache common values
        timestamp = get_wib_timestamp("%H:%M")
        is_armed = self.security_manager.is_armed()

        camera_ids = list(self.cameras.keys())

        for i, camera_id in enumerate(camera_ids):
            if i >= grid_rows * grid_cols:
                break

            # Calculate position once
            row, col = divmod(i, grid_cols)
            y_start, y_end = row * cell_size[1], (row + 1) * cell_size[1]
            x_start, x_end = col * cell_size[0], (col + 1) * cell_size[0]

            camera = self.cameras[camera_id]

            if camera['frame'] is not None and camera['connected']:
                # HD frame processing dengan interpolation terbaik
                frame = cv2.resize(camera['frame'], cell_size, interpolation=cv2.INTER_LANCZOS4)
                overlay = frame.copy()

                # Header bar yang lebih besar
                header_height = int(cell_size[1] * 0.08)  # 8% dari tinggi cell
                cv2.rectangle(overlay, (0, 0), (cell_size[0], header_height), (0, 0, 0), -1)

                # Camera name dengan font yang lebih besar dan crisp
                font_scale = max(0.7, cell_size[0] / 1200)  # Dynamic font scaling
                cv2.putText(overlay, camera['name'], (12, int(header_height * 0.7)),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

                # Status indicators - HD positioning
                indicator_size = max(4, int(cell_size[0] / 200))  # Dynamic indicator size
                status_color = (0, 150, 0) if is_armed else (80, 80, 80)
                cv2.circle(overlay, (cell_size[0] - 20, int(header_height/2)), indicator_size, status_color, -1)

                if camera['alerts']:
                    cv2.circle(overlay, (cell_size[0] - 40, int(header_height/2)), indicator_size, (0, 0, 200), -1)

                # Timestamp - cached dengan font yang lebih besar
                timestamp_font_scale = max(0.4, cell_size[0] / 1500)
                cv2.putText(overlay, timestamp, (cell_size[0] - 100, int(header_height * 0.7)),
                           cv2.FONT_HERSHEY_SIMPLEX, timestamp_font_scale, (200, 200, 200), 2)

                # IR indicator - HD optimized
                ir_color = (0, 255, 0) if camera.get('night_mode', False) else (0, 0, 255)
                ir_y = cell_size[1] - 15
                ir_width = max(45, int(cell_size[0] / 20))
                ir_height = max(20, int(cell_size[1] / 25))
                cv2.rectangle(overlay, (cell_size[0] - ir_width, ir_y - ir_height),
                             (cell_size[0] - 8, ir_y), (0, 0, 0), -1)
                ir_font_scale = max(0.5, cell_size[0] / 1600)
                cv2.putText(overlay, "IR", (cell_size[0] - ir_width + 8, ir_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, ir_font_scale, ir_color, 2)

                # Safe efficient blending
                try:
                    if frame.shape == overlay.shape and frame.dtype == overlay.dtype:
                        frame_with_overlay = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
                    else:
                        frame_with_overlay = frame  # Fallback if shapes don't match
                except Exception as e:
                    logger.debug(f"Grid overlay blend failed: {e}")
                    frame_with_overlay = frame
            else:
                # Optimized disconnected view
                frame_with_overlay = np.full((cell_size[1], cell_size[0], 3), 15, dtype=np.uint8)

                cv2.putText(frame_with_overlay, camera['name'], (8, cell_size[1]//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                cv2.putText(frame_with_overlay, "Connecting...", (8, cell_size[1]//2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)

            # Direct assignment - more efficient
            grid[y_start:y_end, x_start:x_end] = frame_with_overlay

        return grid

    def get_layout_config(self):
        """Get layout configuration - HD quality layouts"""
        layouts = {
            "2x2": {"grid": (2, 2), "size": (960, 540)},  # Upgrade dari 640x360 ke 960x540 (1.5x)
            "1x1": {"grid": (1, 1), "size": (1920, 1080)}  # Upgrade ke Full HD
        }
        return layouts.get(self.current_layout, layouts["2x2"])

    def mouse_callback(self, event, x, y, flags, param):
        """Enhanced mouse callback with security features"""
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.fullscreen_camera is None:
                # Enter fullscreen
                layout = self.get_layout_config()
                grid_rows, grid_cols = layout["grid"]
                cell_width = 1920 // grid_cols  # Update untuk HD resolution
                cell_height = 1080 // grid_rows

                clicked_col = x // cell_width
                clicked_row = y // cell_height
                clicked_index = clicked_row * grid_cols + clicked_col

                camera_ids = list(self.cameras.keys())
                if clicked_index < len(camera_ids):
                    self.fullscreen_camera = camera_ids[clicked_index]
                    logger.info(f"Fullscreen: {self.cameras[self.fullscreen_camera]['name']}")
            else:
                # Exit fullscreen
                self.fullscreen_camera = None

    def display_loop(self):
        """Enhanced display loop with HD quality"""
        cv2.namedWindow('Professional Security System', cv2.WINDOW_NORMAL)
        # Upgrade to Full HD resolution for crisp display
        cv2.resizeWindow('Professional Security System', 1920, 1080)
        cv2.setMouseCallback('Professional Security System', self.mouse_callback)

        while self.running:
            display_frame = self.create_display()

            # HD Status bar untuk grid view
            if self.fullscreen_camera is None:
                status_height = 50  # Tinggi status bar lebih besar
                status_bar = np.zeros((status_height, display_frame.shape[1], 3), dtype=np.uint8)
                status_bar[:] = (30, 30, 30)  # Clean dark background

                # LEFT SECTION - Security Status (Most Important)
                x_pos = 20

                # Security Armed Status dengan font lebih besar
                armed_color = (0, 180, 0) if self.security_manager.is_armed() else (255, 0, 0)
                armed_text = "ARMED" if self.security_manager.is_armed() else "DISARMED"
                cv2.circle(status_bar, (x_pos, 25), 10, armed_color, -1)  # Circle lebih besar
                cv2.putText(status_bar, armed_text, (x_pos + 20, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, armed_color, 2)  # Font lebih besar
                x_pos += 150

                # AI Detection Status dengan font lebih besar
                if self.detection_enabled and self.yolo_detector.detection_enabled:
                    det_color = (0, 180, 0)
                    det_text = "AI ACTIVE"
                else:
                    det_color = (255, 0, 0)
                    det_text = "AI OFF!"

                cv2.circle(status_bar, (x_pos, 25), 10, det_color, -1)  # Circle lebih besar
                cv2.putText(status_bar, det_text, (x_pos + 20, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, det_color, 2)  # Font lebih besar
                x_pos += 130

                # IR indicator removed from status bar - now individual per camera

                # CENTER SECTION - System Info dengan font HD
                center_x = display_frame.shape[1] // 2 - 150

                # Layout Mode dengan font lebih besar
                layout_text = "ALL CAMERAS" if self.current_layout == "2x2" else "SINGLE CAM"
                cv2.putText(status_bar, layout_text, (center_x, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

                # Hardware Mode dengan font lebih besar
                hardware_mode = "GPU" if self.yolo_detector.use_gpu else "CPU"
                hw_color = (0, 150, 0) if self.yolo_detector.use_gpu else (150, 150, 0)
                cv2.putText(status_bar, hardware_mode, (center_x + 150, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, hw_color, 2)

                # Confidence Threshold dengan font lebih besar
                night_cameras = sum(1 for cam in self.cameras.values() if cam.get('night_mode', False))
                if night_cameras > 0:
                    conf_text = f"Conf:{YOLO_CONFIG.get('confidence_threshold_night', 0.65):.2f}"
                    conf_color = (100, 200, 255)
                else:
                    conf_text = f"Conf:{YOLO_CONFIG.get('confidence_threshold', 0.55):.2f}"
                    conf_color = (200, 200, 100)
                cv2.putText(status_bar, conf_text, (center_x + 220, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)

                # RIGHT SECTION - Controls dengan font HD
                right_x = display_frame.shape[1] - 600
                controls = "S:Screenshot | L:View | D:Detection | A:Security | ESC:Exit"
                cv2.putText(status_bar, controls, (right_x, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 2)

                # WARNING - Only if security compromised (overlay on center)
                if not self.security_manager.is_armed() or not (self.detection_enabled and self.yolo_detector.detection_enabled):
                    warning_x = display_frame.shape[1] // 2 - 100
                    cv2.putText(status_bar, "⚠️ SECURITY COMPROMISED", (warning_x - 50, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Combine
                display_frame = np.vstack([status_bar, display_frame])

            cv2.imshow('Professional Security System', display_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Screenshot
                self.take_screenshot()
            elif key == ord('l'):  # Layout
                layouts = ["2x2", "1x1"]
                current_index = layouts.index(self.current_layout)
                self.current_layout = layouts[(current_index + 1) % len(layouts)]
                layout_name = "ALL CAMERAS (2x2)" if self.current_layout == "2x2" else "SINGLE CAMERA (1x1)"
                logger.info(f"View mode: {layout_name}")
            elif key == ord('d'):  # Detection toggle - WITH WARNING
                if self.detection_enabled and self.yolo_detector.detection_enabled:
                    logger.warning("⚠️ HOME SECURITY: Detection is critical for security!")
                    logger.warning("⚠️ Press 'D' again within 3 seconds to confirm disable")

                    # Wait for confirmation
                    confirm_start = time.time()
                    while time.time() - confirm_start < 3:
                        key_confirm = cv2.waitKey(100) & 0xFF
                        if key_confirm == ord('d'):
                            self.yolo_detector.toggle_detection()
                            self.detection_enabled = self.yolo_detector.detection_enabled
                            logger.warning(f"🚨 SECURITY DETECTION {'DISABLED' if not self.detection_enabled else 'ENABLED'}")
                            break
                    else:
                        logger.info("✅ Detection disable cancelled - Security maintained")
                else:
                    # Re-enable detection (no warning needed)
                    self.yolo_detector.detection_enabled = True
                    self.detection_enabled = True
                    logger.info("✅ Detection re-enabled for security")

            elif key == ord('a'):  # Arm/Disarm - WITH WARNING
                if self.security_manager.is_armed():
                    logger.warning("⚠️ HOME SECURITY: Disarming security system!")
                    logger.warning("⚠️ Press 'A' again within 3 seconds to confirm disarm")

                    # Wait for confirmation
                    confirm_start = time.time()
                    while time.time() - confirm_start < 3:
                        key_confirm = cv2.waitKey(100) & 0xFF
                        if key_confirm == ord('a'):
                            armed = self.security_manager.toggle_armed()
                            logger.warning(f"🚨 SECURITY SYSTEM {'ARMED' if armed else 'DISARMED'}")
                            break
                    else:
                        logger.info("✅ Disarm cancelled - Security maintained")
                else:
                    # Re-arm (no warning needed)
                    self.security_manager.armed = True
                    logger.info("✅ Security system re-armed")

            elif key == ord('g'):  # GPU/CPU toggle
                if self.yolo_detector.is_model_loaded():
                    current_gpu = self.yolo_detector.use_gpu
                    self.yolo_detector.switch_device(not current_gpu)
                    self.performance_config = get_active_performance_config()
                    logger.info(f"Switched to {'GPU' if not current_gpu else 'CPU'} mode")
                else:
                    logger.info("YOLO not available for device switch")
            elif key == ord('r'):  # Reconnect
                logger.info("Manual reconnect triggered")
                for camera_id in self.cameras:
                    self.cameras[camera_id]['connected'] = False
            elif key == ord('f'):  # Fullscreen toggle
                if self.fullscreen_camera:
                    self.fullscreen_camera = None
                else:
                    camera_ids = list(self.cameras.keys())
                    if camera_ids:
                        self.fullscreen_camera = camera_ids[0]

        self.stop()

    def start(self):
        """Start the professional security system"""
        self.running = True

        # Start camera threads
        for camera_id in self.cameras:
            thread = threading.Thread(target=self.camera_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
            self.cameras[camera_id]['thread'] = thread

        # Start display
        self.display_loop()

    def stop(self):
        """Stop the security system"""
        logger.info("Stopping Professional Security System...")
        self.running = False

        # Stop security manager (includes cleanup threads)
        self.security_manager.shutdown()

        # Stop YOLO detector
        if hasattr(self.yolo_detector, 'stop_async_detection'):
            self.yolo_detector.stop_async_detection()

        # Close camera connections
        for camera_id, camera in self.cameras.items():
            if camera['cap']:
                camera['cap'].release()

        cv2.destroyAllWindows()
        logger.info("Professional Security System stopped")

def main():
    """Main function"""
    # Create security system
    security_system = ProfessionalSecuritySystem()

    # Add cameras from config
    for config in CAMERAS_CONFIG:
        security_system.add_camera(
            camera_id=config['id'],
            ip=config['ip'],
            username=config['username'],
            password=config['password'],
            channel=config.get('channel', 1),
            name=config.get('name'),
            security_zone=config.get('security_zone', 'unknown'),
            priority=config.get('priority', 'medium')
        )

    try:
        logger.info("🔒 Starting Professional Home Security System...")
        logger.info("====================================================")
        logger.info("🎯 Features: YOLO Detection | Alert System | Recording")
        logger.info("🔧 Controls:")
        logger.info("   ESC/Q: Exit")
        logger.info("   S: Take Screenshot (HD 16:9)")
        logger.info("   L: Switch View (2x2 All Cameras / 1x1 Single)")
        logger.info("   D: Toggle YOLO Detection (Requires Confirmation)")
        logger.info("   A: Arm/Disarm Security (Requires Confirmation)")
        logger.info("   G: Switch GPU/CPU Mode")
        logger.info("   F: Fullscreen Toggle")
        logger.info("   R: Force Reconnect")
        logger.info("   Double-click: Camera Fullscreen")
        logger.info("====================================================")
        logger.info("🏠 HOME SECURITY MODE: Detection & Security Always ON")

        if security_system.detection_enabled:
            stats = security_system.yolo_detector.get_detection_stats()
            hardware_info = security_system.yolo_detector.get_hardware_info()

            logger.info(f"🤖 YOLO Status: Active")
            logger.info(f"   Hardware: {stats['hardware']}")
            logger.info(f"   Device: {stats['device']}")

            if hardware_info.get('name'):
                logger.info(f"   GPU: {hardware_info['name']}")
                logger.info(f"   Memory: {hardware_info.get('memory_total', 0):.1f} GB")

            logger.info(f"   Async: {stats['async_enabled']}")
            logger.info(f"   Performance: {stats['performance_profile'].upper()} optimized")

            # Anti-spam information
            if 'anti_spam' in stats:
                anti_spam = stats['anti_spam']
                logger.info(f"   Anti-Spam: Movement threshold {anti_spam['movement_threshold']}px")
                logger.info(f"   Cooldowns: {anti_spam['spam_cooldown']}s spam, {anti_spam['static_timeout']}s static")
        else:
            logger.info("🤖 YOLO Status: Disabled (install ultralytics for AI detection)")

        security_system.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
    finally:
        security_system.stop()

if __name__ == "__main__":
    main()
