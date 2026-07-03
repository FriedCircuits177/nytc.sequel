"""thanksharshal :)"""

import logging
import queue
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

        # Deadzone threshold
        self.DEADZONE = 0.1

        # Keep track of state to avoid spamming zero tuples endlessly
        self.sent_zero_last = False

    def _result_callback(
        self,
        result: PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Asynchronous callback function where MediaPipe delivers the pose landmarks."""
        if not result or not result.pose_landmarks:
            if not self.sent_zero_last:
                self._push_to_robot_queue(0.0, 0.0, 0.0)
                self.sent_zero_last = True

            with self.shared_state.webcam_draw_data_lock:
                self.shared_state.webcam_draw_data = []
            return

        # Valid tracking frame received
        self.lost_frame_count = 0
        landmarks = result.pose_landmarks[0]

        try:
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]

            # 1. Calculate relative extensions (vertical)
            left_hand_up = l_shoulder.y - l_wrist.y
            right_hand_up = r_shoulder.y - r_wrist.y

            # 2. Evaluate deadzones
            l_active = abs(left_hand_up) > self.DEADZONE
            r_active = abs(right_hand_up) > self.DEADZONE

            # 3. Check for the Crossed Hands gesture inside the deadzone
            # If neither hand is extended vertically, check if they cross horizontally
            hands_crossed = False
            if not l_active and not r_active:
                # Left wrist X is greater than Right wrist X -> they have crossed over
                if l_wrist.x > r_wrist.x:
                    hands_crossed = True

            # 4. Handle data routing conditions based on the gestures
            # if hands_crossed:
            #     # Target action when hands are crossed inside the deadzone
            #     logger.info("Gesture Detected: Hands Crossed inside Deadzone!")
            #     # For example: Trigger a stop command, clear states, or set a specific flag
            #     self._push_to_robot_queue(0.0, 0.0, 0.0)
            #     self.shared_state.phase_state.is_running.clear()

            elif not l_active and not r_active:
                # Normal deadzone behavior (hands up neutral, not crossed)
                if not self.sent_zero_last:
                    self._push_to_robot_queue(0.0, 0.0, 0.0)
                    self.sent_zero_last = True
            else:
                # Driving behavior (active extensions)
                scale = 1.0 / 0.35
                drive_y = ((left_hand_up + right_hand_up) / 2.0) * scale
                drive_r = (right_hand_up - left_hand_up) * scale

                final_y = max(-1.0, min(1.0, drive_y))
                final_r = max(-1.0, min(1.0, drive_r))

                self._push_to_robot_queue(0.0, final_y, final_r)
                self.sent_zero_last = False

            # 4. Compute the mid-point of the shoulders to establish the deadzone center line
            avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0

            # Formulate the top and bottom bounds of the deadzone corridor
            deadzone_top = avg_shoulder_y - self.DEADZONE
            deadzone_bottom = avg_shoulder_y + self.DEADZONE

            # Define Theme Primitives
            COLOR_ACTIVE = (0, 255, 0)  # Bright Green
            COLOR_DEADZONE = (255, 140, 0)  # Amber/Orange
            COLOR_ANCHOR = (0, 255, 255)  # Cyan
            COLOR_BAND = (60, 60, 60)  # Sleek Dark Gray Band for the deadzone overlay

            # Initialize rendering queue array
            new_draw_data = []

            # --- ADD DEADZONE BAND TARGET PRIMITIVE ---
            # Normalized values spanning completely across the screen width (X: 0.0 to 1.0)
            new_draw_data.append(
                {
                    "type": "rectangle",
                    "top_left": (0.0, deadzone_top),
                    "bottom_right": (1.0, deadzone_bottom),
                    "color": COLOR_BAND,
                    "alpha": 0.50,  # 50% opacity target request
                }
            )

            # --- ADD JOINT INDICATORS ---
            tracked_points = [
                {
                    "center": (l_shoulder.x, l_shoulder.y),
                    "color": COLOR_ANCHOR,
                    "radius": 5,
                },
                {
                    "center": (r_shoulder.x, r_shoulder.y),
                    "color": COLOR_ANCHOR,
                    "radius": 5,
                },
                {
                    "center": (l_wrist.x, l_wrist.y),
                    "color": COLOR_ACTIVE if l_active else COLOR_DEADZONE,
                    "radius": 8,
                },
                {
                    "center": (r_wrist.x, r_wrist.y),
                    "color": COLOR_ACTIVE if r_active else COLOR_DEADZONE,
                    "radius": 8,
                },
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
