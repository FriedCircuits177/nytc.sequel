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

    def _result_callback(
        self,
        result: PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Asynchronous callback function where MediaPipe delivers the pose landmarks."""
        if not result or not result.pose_landmarks:
            # If no person detected, safely drift or stop the robot
            self.latest_drive_y = 0.0
            self.latest_drive_r = 0.0
            return

        # MediaPipe provides a list of people detected (we take the first one)
        landmarks = result.pose_landmarks[0]

        # Landmark Indices: Left Shoulder (11), Right Shoulder (12), Left Wrist (15), Right Wrist (16)
        try:
            l_shoulder_y = landmarks[11].y
            r_shoulder_y = landmarks[12].y
            l_wrist_y = landmarks[15].y
            r_wrist_y = landmarks[16].y

            # --- ANALOG CALCULATION ---
            # NOTE: Because image Y decreases as you go UP, (Shoulder Y - Wrist Y) is POSITIVE when hands are raised.
            # Max fully raised extension is typically around 0.3 to 0.5 units in normalized coordinates.

            # 1. Calculate individual relative hand extensions (positive = up, negative = down)
            left_hand_up = l_shoulder_y - l_wrist_y
            right_hand_up = r_shoulder_y - r_wrist_y

            # 2. Scale factor: Map a full arm extension (approx 0.35 normalized units) to a 1.0 motor speed limit
            scale = 1.0 / 0.35

            # Compute analog drives
            # Forward/Backward (drive_y): Average height of both hands
            drive_y = ((left_hand_up + right_hand_up) / 2.0) * scale

            # Rotation (drive_r): Difference between hands.
            # E.g., Right hand UP and Left hand DOWN yields a positive difference -> turns Left.
            drive_r = (right_hand_up - left_hand_up) * scale

            # 3. Clip outputs between -1.0 and 1.0 to ensure safe bounds for the motor controller
            self.latest_drive_y = max(-1.0, min(1.0, drive_y))
            self.latest_drive_r = max(-1.0, min(1.0, drive_r))

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

                    # 6. Apply calculated values safely to your robot state variables
                    # Assuming shared_state properties require basic locking or direct modification
                    self.shared_state.drive_y = self.latest_drive_y
                    self.shared_state.drive_r = self.latest_drive_r
                    logger.info("NO IT WAS ALL GOOD I THINK")

                except Exception as e:
                    logger.error(f"Error in main loop iteration: {e}")

                # Cap execution speed roughly to match standard camera framerates (e.g., ~30 FPS)
                time.sleep(0.03)
