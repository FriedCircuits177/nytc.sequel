import logging
import queue

import cv2
import numpy as np

import ns_shared

logger = logging.getLogger(__name__)


class BlockDetector:
    def __init__(
        self,
        QueueChannels: ns_shared.QueueChannels,  # ns_shared.QueueChannels
        SharedState: ns_shared.SharedState,  # ns_shared.SharedState
    ):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState

        # --- PHYSICAL & CALIBRATION PARAMETERS ---
        self.cube_real_width = 5.0
        self.calibrated_focal_length = 120.61

        # --- TUNING PARAMETERS ---
        self.min_contour_area = 150
        self.morphology_kernel = np.ones((3, 3), np.uint8)

        logger.info("Multi-color BlockDetector initialized successfully")

    def get_latest_frame(self):
        """Safely pulls a local copy of the frame from the shared state."""
        with self.shared_state.eng_camera_frame_lock:
            frame = self.shared_state.eng_camera_frame
        return frame

    def process_frame(self, frame):
        """Performs optimized math tracking loops over the BGR robot frame for both colors and zones."""
        if frame is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_blocks = []
        detected_zones = []  # New list to store zones

        for color_enum in ns_shared.BlockColour:
            lower_bound, upper_bound = color_enum.value
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_contour_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                # --- ZONE CLASSIFICATION (Massive, wide rectangles) ---
                # Adjust these thresholds based on how big the zones look from your camera
                if area > 4000 and aspect_ratio > 1.4:
                    moments = cv2.moments(cnt)
                    if moments["m00"] == 0:
                        continue
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    # NEW TRACKING LINES FOR CLOSEST EDGE
                    # y + h gives the bottom-most pixel row of the contour (closest to robot)
                    bottom_edge_y = y + h

                    # Optional: Convert this bounding width to a physical distance if you know the zone's real width
                    # zone_real_width = 30.0  # (Example: replace with your actual zone width in cm if known)
                    # distance_to_zone = (zone_real_width * self.calibrated_focal_length) / w

                    detected_zones.append(
                        {
                            "color": color_enum,
                            "pixel_center": (cx, cy),
                            "pixel_bounds": (x, y, w, h),
                            "bottom_edge_y": bottom_edge_y,  # Crucial for aligning close to the line
                            "area": area,
                        }
                    )
                    continue

                # --- BLOCK CLASSIFICATION (Small, square cubes) ---
                if not (0.75 <= aspect_ratio <= 1.35):
                    continue

                moments = cv2.moments(cnt)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                distance_z = (self.cube_real_width * self.calibrated_focal_length) / w

                detected_blocks.append(
                    {
                        "color": color_enum,
                        "pixel_center": (cx, cy),
                        "pixel_bounds": (x, y, w, h),
                        "distance_z": distance_z,
                    }
                )

        # Return both structures bundled together
        return {"blocks": detected_blocks, "zones": detected_zones}

    def update_data_queue(self, data):
        """Drops multi-color block payload data into the shared state."""
        with self.shared_state.block_detection_data_lock:
            self.shared_state.block_detection_data = data

    def mainloop(self):
        """Continuously running tracking loop bound to the process kill flag."""
        while not self.queue_channels.kill_flag.is_set():
            self.queue_channels.block_detection_active_flag.wait()
            frame = self.get_latest_frame()
            blocks_data = self.process_frame(frame)
            self.update_data_queue(blocks_data)
