"""thanksharshal :)"""

import logging
import queue
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult

import ns_shared

logger = logging.getLogger(__name__)

model_path = ns_shared.MEDIAPIPE_MODEL_PATH


class MediaPipePoseRecog:
    def __init__(
        self, QueueChannels: ns_shared.QueueChannels, SharedState: ns_shared.SharedState
    ):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState

        # Deadzone threshold
        self.DEADZONE = 0.15

        # Keep track of state to avoid spamming zero tuples endlessly
        self.sent_zero_last = False

        # --- NEW GESTURE DEBOUNCING VARIABLES ---
        self.crossed_hands_counter = 0
        # Require ~15 consecutive frames of holding the cross before triggering (approx 0.5 seconds at 30fps)
        self.CROSSED_HANDS_THRESHOLD = 15
        # Spatial margin to ensure a deliberate cross (normalized coordinates)
        self.CROSS_MARGIN = 0.04

    def _result_callback(
            self,
            result: PoseLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int,
        ):
            if hasattr(self.shared_state, "phase_state") and hasattr(self.shared_state.phase_state, "is_running"):
                if not self.shared_state.phase_state.is_running.is_set():
                    if not self.sent_zero_last:
                        self._push_to_robot_queue(0.0, 0.0, 0.0)
                        self.sent_zero_last = True
                    self.queue_channels.turbo_drive_flag.clear()
                    with self.shared_state.webcam_draw_data_lock:
                        self.shared_state.webcam_draw_data = []
                    return
            """Asynchronous callback function where MediaPipe delivers the pose landmarks."""
            if not result or not result.pose_landmarks:
                if not self.sent_zero_last:
                    self._push_to_robot_queue(0.0, 0.0, 0.0)
                    self.sent_zero_last = True
                self.queue_channels.turbo_drive_flag.clear()

                with self.shared_state.webcam_draw_data_lock:
                    self.shared_state.webcam_draw_data = []
                return

            # Valid tracking frame received
            self.lost_frame_count = 0
            landmarks = result.pose_landmarks[0]

            try:
                # Direct integer indices matching your SDK pattern
                l_shoulder = landmarks[11]
                r_shoulder = landmarks[12]
                l_elbow = landmarks[13]
                r_elbow = landmarks[14]
                l_wrist = landmarks[15]
                r_wrist = landmarks[16]

                # 1. Compute dynamic body-relative vectors (Webcam-tilt proof)
                # Baseline shoulder vector (defines "Horizontal" plane)
                shoulder_vector = np.array([l_shoulder.x - r_shoulder.x, l_shoulder.y - r_shoulder.y])
                shoulder_unit = shoulder_vector / np.linalg.norm(shoulder_vector)

                # Left Arm Vectors (Shoulder -> Elbow -> Wrist)
                l_se_vec = np.array([l_elbow.x - l_shoulder.x, l_elbow.y - l_shoulder.y])
                l_ew_vec = np.array([l_wrist.x - l_elbow.x, l_wrist.y - l_elbow.y])

                # Right Arm Vectors (Shoulder -> Elbow -> Wrist)
                r_se_vec = np.array([r_elbow.x - r_shoulder.x, r_elbow.y - r_shoulder.y])
                r_ew_vec = np.array([r_wrist.x - r_elbow.x, r_wrist.y - r_elbow.y])

                # 2. Check for straight-line alignment (Elbow Locked Outward)
                # Dot product measures how parallel two normalized vectors are (1.0 = perfectly straight line)
                l_straightness = np.dot(l_se_vec / np.linalg.norm(l_se_vec), l_ew_vec / np.linalg.norm(l_ew_vec))
                r_straightness = np.dot(r_se_vec / np.linalg.norm(r_se_vec), r_ew_vec / np.linalg.norm(r_ew_vec))

                # 3. Check for Horizontality relative to the shoulders
                l_horizontality = abs(np.dot(l_se_vec / np.linalg.norm(l_se_vec), shoulder_unit))
                r_horizontality = abs(np.dot(r_se_vec / np.linalg.norm(r_se_vec), shoulder_unit))

                # 4. Enforce directional safety constraints (Hands must be outside elbows, elbows outside shoulders)
                is_extended_left = (l_wrist.x > l_elbow.x) and (l_elbow.x > l_shoulder.x)
                is_extended_right = (r_wrist.x < r_elbow.x) and (r_elbow.x < r_shoulder.x)

                # Combine checks to declare a valid T-Pose (Allowing a safe 5-10 degree tolerance window)
                l_t_pose = (l_straightness > 0.9) and (l_horizontality > 0.9) and is_extended_left
                r_t_pose = (r_straightness > 0.9) and (r_horizontality > 0.9) and is_extended_right

                t_pose_instant = l_t_pose and r_t_pose

                # 5. Calculate relative extensions (vertical) for standard driving
                left_hand_up = l_shoulder.y - l_wrist.y
                right_hand_up = r_shoulder.y - r_wrist.y

                l_active = abs(left_hand_up) > self.DEADZONE
                r_active = abs(right_hand_up) > self.DEADZONE

                # 6. Robust Crossed Hands Evaluation
                hands_crossed_instantly = False
                if not l_active and not r_active:
                    if r_wrist.x > (l_wrist.x + self.CROSS_MARGIN):
                        hands_crossed_instantly = True

                # Temporal filtering: Require the posture to be held over multiple frames
                if hands_crossed_instantly:
                    self.crossed_hands_counter += 1
                else:
                    self.crossed_hands_counter = 0

                # 7. Core Command Routing
                if self.crossed_hands_counter >= self.CROSSED_HANDS_THRESHOLD:
                    logger.info("Gesture Confirmed: Hands Crossed! Initiating Exit Sequence.")
                    self._push_to_robot_queue(0.0, 0.0, 0.0)
                    self.queue_channels.turbo_drive_flag.clear()

                    if hasattr(self.shared_state, "phase_state") and hasattr(self.shared_state.phase_state, "is_running"):
                        self.shared_state.phase_state.is_running.clear()
                    with self.shared_state.webcam_draw_data_lock:
                        self.shared_state.webcam_draw_data = []
                    return

                elif t_pose_instant:
                    # Trigger dynamic Turbo Mode vector overrides
                    self.queue_channels.turbo_drive_flag.set()
                    self._push_to_robot_queue(0.0, 1.0, 0.0)  # Locked straight max forward vector
                    self.sent_zero_last = False

                elif hands_crossed_instantly or (not l_active and not r_active):
                    if not self.sent_zero_last:
                        self._push_to_robot_queue(0.0, 0.0, 0.0)
                        self.sent_zero_last = True
                    self.queue_channels.turbo_drive_flag.clear()

                else:
                    # Normal Driving Loop
                    self.queue_channels.turbo_drive_flag.clear()
                    scale = 1.0 / 0.35
                    drive_y = ((left_hand_up + right_hand_up) / 2.0) * scale
                    drive_r = (right_hand_up - left_hand_up) * scale

                    final_y = max(-1.0, min(1.0, drive_y))
                    final_r = max(-1.0, min(1.0, drive_r))

                    self._push_to_robot_queue(0.0, final_y, final_r)
                    self.sent_zero_last = False

                # 8. Dynamic UI Color-Feedback Setup
                avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
                deadzone_top = avg_shoulder_y - self.DEADZONE
                deadzone_bottom = avg_shoulder_y + self.DEADZONE

                if hands_crossed_instantly:
                    COLOR_BAND = (0, 0, 255)      # Pure Red (BGR: 0, 0, 255)
                    COLOR_ANCHOR = (0, 0, 255)
                    COLOR_ACTIVE = (0, 0, 255)
                    COLOR_DEADZONE = (0, 0, 255)
                elif t_pose_instant:
                    COLOR_ACTIVE = (0, 242, 255)   # Neon Gold / Yellow
                    COLOR_DEADZONE = (255, 0, 255) # Bright Magenta
                    COLOR_ANCHOR = (0, 255, 0)     # Pure Green
                    COLOR_BAND = (30, 0, 30)       # Dark Purple Hue
                else:
                    COLOR_ACTIVE = (0, 255, 0)     # Bright Green (BGR: 0, 255, 0)
                    COLOR_DEADZONE = (0, 140, 255) # Deep Amber/Orange (BGR: 0, 140, 255)
                    COLOR_ANCHOR = (255, 255, 0)   # Cyan (BGR: 255, 255, 0)
                    COLOR_BAND = (60, 60, 60)      # Sleek Dark Gray Band
                # Build data packing format matching your original dictionary drawing format
                new_draw_data = []
                new_draw_data.append(
                    {
                        "type": "rectangle",
                        "top_left": (0.0, deadzone_top),
                        "bottom_right": (1.0, deadzone_bottom),
                        "color": COLOR_BAND,
                        "alpha": 0.50,
                        "thickness": -1,
                    }
                )

                tracked_points = [
                    {"center": (l_shoulder.x, l_shoulder.y), "color": COLOR_ANCHOR, "radius": 5},
                    {"center": (r_shoulder.x, r_shoulder.y), "color": COLOR_ANCHOR, "radius": 5},
                    {"center": (l_wrist.x, l_wrist.y), "color": COLOR_ACTIVE if l_active or t_pose_instant else COLOR_DEADZONE, "radius": 8},
                    {"center": (r_wrist.x, r_wrist.y), "color": COLOR_ACTIVE if r_active or t_pose_instant else COLOR_DEADZONE, "radius": 8},
                ]

                for pt in tracked_points:
                    new_draw_data.append(
                        {
                            "type": "circle",
                            "center": pt["center"],
                            "color": pt["color"],
                            "radius": pt["radius"],
                            "thickness": -1,
                        }
                    )

                with self.shared_state.webcam_draw_data_lock:
                    self.shared_state.webcam_draw_data = new_draw_data

            except Exception as e:
                logger.error(f"Error extracting landmarks: {e}")

    def _push_to_robot_queue(self, x, y, r):
        """Pushes data onto the thread-safe queue cleanly without blocking the main stream thread."""
        try:
            # Clear old frame if robot fell slightly behind to prevent lag buildup
            while True:
                self.queue_channels.pose_drive.get_nowait()
        except queue.Empty:
            pass

        try:
            self.queue_channels.pose_drive.put_nowait((x, y, r))
        except queue.Full:
            pass

    def mainloop(self):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
        )

        with vision.PoseLandmarker.create_from_options(options) as detector:
            logger.info("MediaPipe Pose Landmarker successfully initialized.")

            while not self.queue_channels.kill_flag.is_set():
                frame = None
                self.queue_channels.pose_recog_active_flag.wait()

                if hasattr(self.shared_state, "phase_state") and hasattr(self.shared_state.phase_state, "is_running"):
                    if not self.shared_state.phase_state.is_running.is_set():
                        with self.shared_state.webcam_draw_data_lock:
                            self.shared_state.webcam_draw_data = []
                        time.sleep(0.05)
                        continue

                with self.shared_state.raw_webcam_camera_frame_lock:
                    if self.shared_state.raw_webcam_camera_frame is not None:
                        frame = self.shared_state.raw_webcam_camera_frame.copy()
                if frame is None:
                    time.sleep(0.01)
                    continue

                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb_frame
                    )

                    timestamp_ms = int(time.time() * 1000)
                    detector.detect_async(mp_image, timestamp_ms)

                except Exception as e:
                    logger.error(f"Error in main loop iteration: {e}")

                time.sleep(0.03)
