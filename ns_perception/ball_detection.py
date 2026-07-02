import logging
import time

import cv2
import numpy as np

import ns_shared

logger = logging.getLogger(__name__)


class BallDetector:
    def __init__(
        self, QueueChannels: ns_shared.QueueChannels, SharedState: ns_shared.SharedState
    ):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.KNOWN_WIDTH = 3  # diameter of the ball
        self.FOCAL_LENGTH = ns_shared.CAMERA_FOCAL_LENGTH

    def get_ball_data(self, frame):
        """
        Processes a frame to find a red ball.
        Returns: (x_error, distance, normalized_x) or (None, None, None)
        """
        if frame is None:
            return None, None, None

        # 1. Blur to reduce high-frequency noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. Red color masking (handles the HSV wrap-around)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # 4. Clean up the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 5. Find contours
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            # Find the largest contour (assumed to be the ball)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

            # Filter out tiny noise blobs
            if radius > 10:
                frame_width = frame.shape[1]
                frame_center_x = frame_width / 2.0

                # --- X Position & Centering Error ---
                # Raw pixel error from the center of the screen
                x_error = x - frame_center_x
                # Normalized error between -1.0 and 1.0 (great for feeding into motor formulas)
                normalized_x = x_error / frame_center_x

                # --- Distance Estimation ---
                # Using the triangle similarity theorem: Distance = (TrueWidth * FocalLength) / PixelWidth
                pixel_width = radius * 2
                distance = (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / pixel_width

                return {
                    "x_error": x_error,
                    "distance": distance,
                    "normalized_x": normalized_x,
                    "y": y,
                }

        return {}

    def mainloop(self):
        while not self.queue_channels.kill_flag.is_set():
            self.queue_channels.ball_detection_active_flag.wait()  # block until needed
            with self.shared_state.eng_camera_frame_lock:
                frame = self.shared_state.eng_camera_frame
            data = self.get_ball_data(frame)
            with self.shared_state.ball_detection_data_lock:
                self.shared_state.ball_detection_data = data
            time.sleep(0.05)
