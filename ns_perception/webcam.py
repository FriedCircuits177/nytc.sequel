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

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            raise RuntimeError("Camera failed to open")
        logger.info("Initialised succesfully")
        # self.temp_counter = 0

    def poll_camera_frame(self):
        # self.temp_counter += 1

        self.capture.grab()  # fast: just grab latest frame
        ret, frame = self.capture.retrieve()

        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def put_camera_frame(self, frame):
        with self.camera_frame_lock:
            self.camera_frame = frame
            if self.camera_frame_lock is self.shared_state.raw_webcam_camera_frame_lock:
                self.shared_state.raw_webcam_camera_frame = frame
            elif self.camera_frame_lock is self.shared_state.webcam_camera_frame_lock:
                self.shared_state.webcam_camera_frame = frame

    def mainloop(self):
        while not self.queue_channels.kill_flag.is_set():
            self.put_camera_frame(self.poll_camera_frame())
            # time.sleep(0.001)


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

        self.fps_frames = 0
        self.fps_last_time = time.time()
        self.fps_count = 0
        self.fps_sum = 0.0
        self.max_fps = 0
        self.fps_text = "FPS: 0 AVG: 0 MAX: 0"
        self.last_active_state = False

    def process(self, frame):
        if frame is None:
            return None

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        # --- NEW OVERLAY RENDERING BLOCK ---
        # Fetch a local snapshot of the pose draw primitives safely
        with self.shared_state.webcam_draw_data_lock:
            draw_primitives = list(self.shared_state.webcam_draw_data)

        # Draw each primitive onto the frame before converting it for the GUI
        for primitive in draw_primitives:
            p_type = primitive.get("type")

            if p_type == "rectangle":
                # Convert normalized coordinates (0.0 to 1.0) back to actual pixel coordinates
                top_left_raw = primitive["top_left"]
                bottom_right_raw = primitive["bottom_right"]
                pt1 = (
                    int(top_left_raw[0] * self.width),
                    int(top_left_raw[1] * self.height),
                )
                pt2 = (
                    int(bottom_right_raw[0] * self.width),
                    int(bottom_right_raw[1] * self.height),
                )

                color = primitive[
                    "color"
                ]  # BGR or RGB depending on webcam frame format
                thickness = primitive["thickness"]

                # Check for alpha opacity request
                alpha_val = primitive.get("alpha", 1.0)
                if alpha_val < 1.0:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, pt1, pt2, color, thickness)
                    cv2.addWeighted(
                        overlay, alpha_val, frame, 1.0 - alpha_val, 0, frame
                    )
                else:
                    cv2.rectangle(frame, pt1, pt2, color, thickness)

            elif p_type == "circle":
                center_raw = primitive["center"]
                center = (
                    int(center_raw[0] * self.width),
                    int(center_raw[1] * self.height),
                )
                radius = primitive["radius"]
                color = primitive["color"]
                thickness = primitive["thickness"]

                cv2.circle(frame, center, radius, color, thickness)
        # ------------------------------------

        # Original logic: Normalise and push to DearPyGUI texture format
        # --- FPS CALCULATOR LAYER ---
        self.fps_frames += 1
        now = time.time()
        if now - self.fps_last_time >= 1.0:
            current_fps = self.fps_frames
            self.fps_frames = 0
            self.fps_count += 1
            self.fps_sum += current_fps
            mean_fps = self.fps_sum / self.fps_count
            if current_fps > self.max_fps:
                self.max_fps = current_fps

            self.fps_text = (
                f"FPS: {current_fps} AVG: {int(mean_fps)} MAX: {self.max_fps}"
            )
            self.fps_last_time = now

        # Burn FPS text with a dark high-contrast shadow offset by 1 pixel
        # cv2.putText(frame, self.fps_text, (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            self.fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Original logic: Normalise and push to DearPyGUI texture format
        np.divide(frame, 255.0, out=self.output[:, :, :3], casting="unsafe")
        self.output[:, :, 3] = 1.0
        self.output = cv2.flip(self.output, 1)

        return self.output.copy()

    def mainloop(self):
        last_frame_id = None

        while not self.queue_channels.kill_flag.is_set():
            with self.raw_camera_frame_lock:
                if (
                    self.raw_camera_frame_lock
                    is self.shared_state.raw_webcam_camera_frame_lock
                ):
                    frame = self.shared_state.raw_webcam_camera_frame
                else:
                    frame = self.raw_camera_frame

            if frame is None:
                time.sleep(0.001)
                continue

            # CRITICAL FIX: Make a unique copy of the frame array!
            # Since OpenCV operations above (cv2.rectangle/circle) modify 'frame' IN-PLACE,
            # the underlying object id(frame) would remain the same, causing this loop to
            # falsely skip frames thinking it's the exact same stale data.
            frame_copy = frame.copy()

            if id(frame) == last_frame_id:
                # If the frame hasn't updated, we still check if overlays have changed
                # to keep rendering smooth.
                pass

            last_frame_id = id(frame)

            processed = self.process(frame_copy)

            with self.camera_frame_lock:
                self.camera_frame = processed
                if self.camera_frame_lock is self.shared_state.webcam_camera_frame_lock:
                    self.shared_state.webcam_camera_frame = processed

            time.sleep(0.01)  # Keep CPU load nominal
