import logging
import time

import cv2
import numpy as np

import ns_shared

logger = logging.getLogger(__name__)


class Webcam:
    def __init__(
        self,
        QueueChannels: ns_shared.QueueChannels,
        SharedState: ns_shared.SharedState,
        camera_frame,
        camera_frame_lock,
    ):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.camera_frame = camera_frame
        self.camera_frame_lock = camera_frame_lock

        # 1. Force DirectShow backend
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # 2. CRITICAL: Force the driver to use MJPEG decompression.
        # This bypasses uncompressed format locking issues common in background threads.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # 3. Apply resolution settings
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.capture.isOpened():
            raise RuntimeError("Camera failed to open")
        logger.info("Initialised successfully via DirectShow MJPEG")

    def poll_camera_frame(self):
        # FIX: Use integrated read() execution path instead of discrete grab/retrieve split
        ret, frame = self.capture.read()

        if not ret or frame is None:
            return None

        frame = cv2.flip(frame, 1)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def put_camera_frame(self, frame):
        with self.camera_frame_lock:
            self.camera_frame = frame
            if self.camera_frame_lock is self.shared_state.raw_webcam_camera_frame_lock:
                self.shared_state.raw_webcam_camera_frame = frame
            elif self.camera_frame_lock is self.shared_state.webcam_camera_frame_lock:
                self.shared_state.webcam_camera_frame = frame

    def mainloop(self):
        while not self.queue_channels.kill_flag.is_set():
            # logger.info("webcam running")
            frame = self.poll_camera_frame()

            if frame is not None:
                self.put_camera_frame(frame)

            # Slightly longer yield window to allow driver resource swapping
            # to occur smoothly between frames
            time.sleep(0.01)


class WebcamProcessor:
    """Thine only purpose is to make the raw frame usable by DearPyGUI."""

    def __init__(
        self,
        QueueChannels: ns_shared.QueueChannels,
        SharedState: ns_shared.SharedState,
        raw_camera_frame,
        raw_camera_frame_lock,
        camera_frame,
        camera_frame_lock,
    ):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.camera_frame = camera_frame
        self.camera_frame_lock = camera_frame_lock
        self.raw_camera_frame = raw_camera_frame
        self.raw_camera_frame_lock = raw_camera_frame_lock

        self.width = 640
        self.height = 480

        self.alpha = np.ones((self.height, self.width), dtype=np.float32)
        self.output = np.empty((self.height, self.width, 4), dtype=np.float32)

    def process(self, frame):
        if frame is None:
            return None

        # Ensure correct frame dimensions
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        else:
            frame = frame.copy()

        # FIX: Safer drawing layer lookup mapping to the correct lock
        draw_items = []
        if (
            hasattr(self.shared_state, "webcam_draw_data")
            and self.shared_state.webcam_draw_data_lock
        ):
            with self.shared_state.webcam_draw_data_lock:
                if self.shared_state.webcam_draw_data is not None:
                    draw_items = list(self.shared_state.webcam_draw_data)

        # --- UNIFIED DRAWING LAYER SYSTEM ---
        for item in draw_items:
            item_type = item.get("type", "circle")
            color = item.get("color", (0, 255, 0))
            thickness = item.get("thickness", 2)

            if item_type == "rectangle":
                if "top_left" in item and "bottom_right" in item:
                    tl_x, tl_y = item["top_left"]
                    br_x, br_y = item["bottom_right"]
                    p1 = (int(tl_x * self.width), int(tl_y * self.height))
                    p2 = (int(br_x * self.width), int(br_y * self.height))
                elif "corners" in item:
                    corners = item["corners"]
                    p1 = (
                        int(corners[0][0] * self.width),
                        int(corners[0][1] * self.height),
                    )
                    p2 = (
                        int(corners[2][0] * self.width),
                        int(corners[2][1] * self.height),
                    )
                else:
                    continue

                if "alpha" in item:
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        p1,
                        p2,
                        color,
                        thickness=-1 if thickness == -1 else thickness,
                    )
                    alpha = item["alpha"]
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
                else:
                    cv2.rectangle(frame, p1, p2, color, thickness)

            elif item_type == "circle":
                norm_x, norm_y = item["center"]
                pixel_x = int(norm_x * self.width)
                pixel_y = int(norm_y * self.height)
                radius = item.get("radius", 6)
                cv2.circle(frame, (pixel_x, pixel_y), radius, color, thickness)

            elif item_type == "text":
                norm_x, norm_y = item["position"]
                pixel_x = int(norm_x * self.width)
                pixel_y = int(norm_y * self.height)
                text_str = item.get("text", "")
                font_scale = item.get("scale", 0.5)

                # Text drop shadow
                cv2.putText(
                    frame,
                    text_str,
                    (pixel_x, pixel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    text_str,
                    (pixel_x, pixel_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        # Convert to DearPyGUI texture format (Normalized float32 RGBA)
        float_frame = frame.astype(np.float32)
        np.divide(float_frame, 255.0, out=self.output[:, :, :3])
        self.output[:, :, 3] = 1.0

        return self.output.ravel()

    def mainloop(self):
        # FIX: Avoid comparing object IDs directly across dynamic thread mutations
        # Track by capturing an internal deep-copy signature or basic loop cadence if tracking mutations fails
        while not self.queue_channels.kill_flag.is_set():
            frame = None
            with self.raw_camera_frame_lock:
                if (
                    self.raw_camera_frame_lock
                    is self.shared_state.raw_webcam_camera_frame_lock
                ):
                    if self.shared_state.raw_webcam_camera_frame is not None:
                        frame = self.shared_state.raw_webcam_camera_frame.copy()
                else:
                    if self.raw_camera_frame is not None:
                        frame = self.raw_camera_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            processed = self.process(frame)

            if processed is not None:
                with self.camera_frame_lock:
                    self.camera_frame = processed
                    if (
                        self.camera_frame_lock
                        is self.shared_state.webcam_camera_frame_lock
                    ):
                        self.shared_state.webcam_camera_frame = processed

            # Match frame cadence to frame rate target (~30 FPS)
            time.sleep(0.03)
