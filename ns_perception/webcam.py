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

        # 1. Enforce sizing constraints and copy source matrix
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        else:
            frame = frame.copy()

        # 2. Fetch the rendering primitives under lock safely
        draw_items = []
        if hasattr(self.shared_state, "pose_draw_data"):
            with self.shared_state.pose_draw_data_lock:
                draw_items = list(self.shared_state.pose_draw_data)

        # 3. Iterate through queue instructions to render types dynamically
        for item in draw_items:
            item_type = item.get(
                "type", "circle"
            )  # Default to circle if type is missing

            if item_type == "rectangle":
                # Extract and scale normalized locations to pixel arrays
                tl_x, tl_y = item["top_left"]
                br_x, br_y = item["bottom_right"]

                # Clip limits safely inside image resolution bounds
                p1 = (int(tl_x * self.width), max(0, int(tl_y * self.height)))
                p2 = (int(br_x * self.width), min(self.height, int(br_y * self.height)))

                # Create a temporary copy layer to apply alpha transparency weight blends
                overlay = frame.copy()
                cv2.rectangle(overlay, p1, p2, color=item["color"], thickness=-1)

                # Apply 50% blend layout handoff
                alpha = item.get("alpha", 0.5)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

            elif item_type == "circle":
                norm_x, norm_y = item["center"]
                pixel_x = int(norm_x * self.width)
                pixel_y = int(norm_y * self.height)

                cv2.circle(
                    frame,
                    (pixel_x, pixel_y),
                    radius=item["radius"],
                    color=item["color"],
                    thickness=item.get("thickness", -1),
                )

        # 4. Transcribe values onto float32 canvas array normalized for DPG
        float_frame = frame.astype(np.float32)
        np.divide(float_frame, 255.0, out=self.output[:, :, :3])
        self.output[:, :, 3] = 1.0

        return self.output.ravel()

    def mainloop(self):
        last_frame_id = None

        while not self.queue_channels.kill_flag.is_set():
            # logging.info("webcam processor is running")
            with self.raw_camera_frame_lock:
                if (
                    self.raw_camera_frame_lock
                    is self.shared_state.raw_webcam_camera_frame_lock
                ):
                    frame = self.shared_state.raw_webcam_camera_frame
                else:
                    frame = self.raw_camera_frame

            # If there is no frame data yet, wait for the capture thread
            if frame is None:
                time.sleep(0.005)
                continue

            if id(frame) == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = id(frame)
            processed = self.process(frame)

            # Only update the final output channel if processing succeeded
            if processed is not None:
                with self.camera_frame_lock:
                    self.camera_frame = processed
                    if (
                        self.camera_frame_lock
                        is self.shared_state.webcam_camera_frame_lock
                    ):
                        self.shared_state.webcam_camera_frame = processed
