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

        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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

    def process(self, frame):
        if frame is None:
            return None

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        np.divide(frame, 255.0, out=self.output[:, :, :3], casting="unsafe")

        self.output[:, :, 3] = 1.0

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

            if id(frame) == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = id(frame)

            processed = self.process(frame)

            with self.camera_frame_lock:
                self.camera_frame = processed
                if self.camera_frame_lock is self.shared_state.webcam_camera_frame_lock:
                    self.shared_state.webcam_camera_frame = processed
