"""thanksharshal :)"""

import logging
import time

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

        # Store temporary outputs from the async callback
        self.latest_drive_y = 0.0
        self.latest_drive_r = 0.0

        # --- TELEMETRY SMOOTHING & DEADZONE CONFIG ---
        self.lost_frame_count = 0
        self.MAX_LOST_FRAMES = 5  # Bridges ~150ms of micro-drops without dropping to 0

        # Deadzone threshold in normalized coordinate units (Y-axis distance)
        self.DEADZONE = (
            0.05  # Hands must be at least this far above/below shoulders to drive
        )

    def _result_callback(
        self,
        result: PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Asynchronous callback function where MediaPipe delivers the pose landmarks."""
        if not result or not result.pose_landmarks:
            self.lost_frame_count += 1

            # Only zero-out the motors if tracking has been completely lost over multiple frames
            if self.lost_frame_count > self.MAX_LOST_FRAMES:
                with self.shared_state.drive_command_lock:
                    self.shared_state.drive_y = 0.0
                    self.shared_state.drive_r = 0.0
                with self.shared_state.pose_draw_data_lock:
                    self.shared_state.pose_draw_data = []
            return

        # Valid frame received, reset our tracking drop counter
        self.lost_frame_count = 0
        landmarks = result.pose_landmarks[0]

        try:
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]

            # 1. Calculate relative arm extensions (Positive = Raised, Negative = Dropped)
            left_hand_up = l_shoulder.y - l_wrist.y
            right_hand_up = r_shoulder.y - r_wrist.y

            # 2. Evaluate deadzone status (Are the hands moved far enough from neutral shoulder height?)
            l_active = abs(left_hand_up) > self.DEADZONE
            r_active = abs(right_hand_up) > self.DEADZONE

            # Filter inputs based on deadzone evaluations
            left_final = left_hand_up if l_active else 0.0
            right_final = right_hand_up if r_active else 0.0

            # 3. Scale factor and analog drive mix
            scale = 1.0 / 0.35
            drive_y = ((left_final + right_final) / 2.0) * scale
            drive_r = (right_final - left_final) * scale

            # 4. Safe bound clipping and direct assignment to global telemetry channels
            with self.shared_state.drive_command_lock:
                self.shared_state.drive_y = max(-1.0, min(1.0, drive_y))
                self.shared_state.drive_r = max(-1.0, min(1.0, drive_r))

            # --- DYNAMIC GUI COLOR CODE FEEDBACK ---
            COLOR_ACTIVE = (0, 255, 0)  # Bright Green: Actively driving motors
            COLOR_DEADZONE = (
                255,
                140,
                0,
            )  # Amber/Orange: Tracked but idling in deadzone
            COLOR_ANCHOR = (0, 255, 255)  # Cyan: Static shoulder reference joints

            new_draw_data = [
                {
                    "center": (l_shoulder.x, l_shoulder.y),
                    "color": COLOR_ANCHOR,
                    "radius": 5,
                    "thickness": -1,
                },
                {
                    "center": (r_shoulder.x, r_shoulder.y),
                    "color": COLOR_ANCHOR,
                    "radius": 5,
                    "thickness": -1,
                },
                {
                    "center": (l_wrist.x, l_wrist.y),
                    "color": COLOR_ACTIVE if l_active else COLOR_DEADZONE,
                    "radius": 8,
                    "thickness": -1,
                },
                {
                    "center": (r_wrist.x, r_wrist.y),
                    "color": COLOR_ACTIVE if r_active else COLOR_DEADZONE,
                    "radius": 8,
                    "thickness": -1,
                },
            ]

            with self.shared_state.pose_draw_data_lock:
                self.shared_state.pose_draw_data = new_draw_data

        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")

    def mainloop(self):
        # 1. Set up options with the Async Callback
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
        )

        # 2. Initialize the detector OUTSIDE the loop for speed
        with vision.PoseLandmarker.create_from_options(options) as detector:
            logger.info("MediaPipe Pose Landmarker successfully initialized.")

            while not self.queue_channels.kill_flag.is_set():
                frame = None
                self.queue_channels.pose_recog_active_flag.wait()
                # logger.info("RUNNING POSE RECOG (I THINK)")
                # 3. Thread-safe frame extraction
                with self.shared_state.raw_webcam_camera_frame_lock:
                    if self.shared_state.raw_webcam_camera_frame is not None:
                        # Make a shallow or deep copy depending on your framework's design
                        frame = self.shared_state.raw_webcam_camera_frame.copy()

                if frame is None:
                    time.sleep(
                        0.01
                    )  # Avoid burning CPU cycles if the camera is lagging
                    logger.info("BUT THE FRAME IS NONE")
                    continue

                try:
                    # 4. Process frame (Convert BGR to RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb_frame
                    )

                    # 5. Send frame to the async tracker with a unique millisecond timestamp
                    timestamp_ms = int(time.time() * 1000)
                    detector.detect_async(mp_image, timestamp_ms)

                    # 6. Cleaned: Handled asynchronously by the callback now!
                    # (Removed self.shared_state.drive_y assignments from here)

                except Exception as e:
                    logger.error(f"Error in main loop iteration: {e}")

                # Cap execution speed roughly to match standard camera framerates (e.g., ~30 FPS)
                time.sleep(0.03)
