import base64
import logging
import time

import cv2
import numpy as np
from turbojpeg import TJPF_BGR, TurboJPEG

import ns_robot
import ns_shared

logger = logging.getLogger(__name__)


class Camera:
    def __init__(
        self,
        robot: ns_robot.RobotHardware,
        queue_channels: ns_shared.QueueChannels,
        shared_state: ns_shared.SharedState,
        camera_frame,
        camera_frame_lock,
        active_flag,
    ):
        self.robot = robot
        self.queue_channels = queue_channels
        self.shared_state = shared_state
        self.camera_frame = camera_frame
        self.camera_frame_lock = camera_frame_lock
        self.active_flag = active_flag
        # Initialize TurboJPEG decoder
        self.tj = TurboJPEG(ns_shared.TURBOJPEG_PATH)

    def readEncodedFrame(self):
        """Reads raw base64 encoded JPEG frame from the robot camera via gRPC and decodes it"""
        try:
            # Reads from low-level unary_unary channel
            return self.robot._sdk.read_camera_data()
        except Exception as e:
            logger.error(f"Failed to read camera data from robot: {e}")
            return ""

    def b64_to_bgr_turbo(self, jpg_bytes) -> np.ndarray | None:
        """Converts raw base64 encoded JPEG into a standard OpenCV BGR numpy array."""
        # if not b64_string:
        #     return None
        try:
            # jpg_bytes = base64.b64decode(b64_string)
            # Directly decode to standard BGR array for your vision perception code
            bgr_frame = self.tj.decode(jpg_bytes, pixel_format=TJPF_BGR)
            return bgr_frame
        except Exception as e:
            logger.error(f"TurboJPEG decoding error: {e}")
            return None

    def put_camera_frame(self, frame):
        """Safely binds the local frame to the matching global SharedState field."""
        if frame is None:
            return

        with self.camera_frame_lock:
            self.camera_frame = frame
            # Identity routing engine to map instances back to true variables
            if self.camera_frame_lock is self.shared_state.sb_camera_frame_lock:
                self.shared_state.sb_camera_frame = frame
            elif self.camera_frame_lock is self.shared_state.eng_camera_frame_lock:
                self.shared_state.eng_camera_frame = frame

    def mainloop(self):
        """Thread loop pulling robot frames and pushing them out as raw BGR frames."""
        logger.info("Robot Camera ingestion loop started.")
        while not self.queue_channels.kill_flag.is_set():
            try:
                self.robot._sdk.open_camera()
                break
            except Exception as e:
                # logging.error(e)
                continue
        while not self.queue_channels.kill_flag.is_set():
            self.active_flag.wait()
            b64_data = self.readEncodedFrame()
            if b64_data:
                bgr_frame = self.b64_to_bgr_turbo(b64_data)
                self.put_camera_frame(bgr_frame)
            else:
                # print("IS NOT b64 DATA")
                # Avoid aggressive spinning if the stream drops frames momentarily
                time.sleep(0.001)


class CameraGUIProcessor:
    """Consumes the perception BGR frames and normalizes them into Float32 RGBA for DearPyGUI."""

    def __init__(
        self,
        queue_channels: ns_shared.QueueChannels,
        shared_state: ns_shared.SharedState,
        raw_camera_frame,
        raw_camera_frame_lock,
        gui_camera_frame,
        gui_camera_frame_lock,
        active_flag,
    ):
        self.queue_channels = queue_channels
        self.shared_state = shared_state
        self.raw_camera_frame = raw_camera_frame
        self.raw_camera_frame_lock = raw_camera_frame_lock
        self.gui_camera_frame = gui_camera_frame
        self.gui_camera_frame_lock = gui_camera_frame_lock
        self.active_flag = active_flag
        # DearPyGUI standard viewport resolution configuration
        self.width = 640
        self.height = 480
        self.output = np.empty((self.height, self.width, 4), dtype=np.float32)

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        if frame is None:
            return None

        # Ensure correct texture sizing
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        # Convert robot BGR to RGB (let OpenCV allocate a temporary contiguous array)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Now safely divide directly into our pre-allocated float32 slice
        np.divide(rgb_frame, 255.0, out=self.output[:, :, :3])

        # Set solid alpha layer
        self.output[:, :, 3] = 1.0

        return self.output.copy()

    def mainloop(self):
        """Thread loop pulling BGR frames, transforming them, and piping to the GUI."""
        last_frame_id = None
        logger.info("Camera GUI conversion processing loop started.")

        while not self.queue_channels.kill_flag.is_set():
            # Safely fetch the latest raw frame depending on the assigned lock
            self.active_flag.wait()
            with self.raw_camera_frame_lock:
                if self.raw_camera_frame_lock is self.shared_state.sb_camera_frame_lock:
                    frame = self.shared_state.sb_camera_frame
                elif (
                    self.raw_camera_frame_lock
                    is self.shared_state.eng_camera_frame_lock
                ):
                    frame = self.shared_state.eng_camera_frame
                else:
                    frame = self.raw_camera_frame

            if frame is None:
                time.sleep(0.001)
                continue

            # Skip execution if the frame object hasn't rotated yet
            if id(frame) == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = id(frame)
            processed_gui_frame = self.process(frame)

            if processed_gui_frame is not None:
                with self.gui_camera_frame_lock:
                    self.gui_camera_frame = processed_gui_frame
                    # Update global shared state based on matching lock target
                    if (
                        self.gui_camera_frame_lock
                        is self.shared_state.sb_gui_camera_frame_lock
                    ):
                        self.shared_state.sb_gui_camera_frame = processed_gui_frame
                    elif (
                        self.gui_camera_frame_lock
                        is self.shared_state.eng_gui_camera_frame_lock
                    ):
                        self.shared_state.eng_gui_camera_frame = processed_gui_frame
