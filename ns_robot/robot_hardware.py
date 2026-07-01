import logging
import time

import cv2
import numpy as np
from ugot import ugot

import ns_shared

logger = logging.getLogger(__name__)


class RobotHardware:
    """Wrapper for the ugot.UGOT instance to allow for custom methods."""

    def __init__(
        self,
        queue_channels: ns_shared.QueueChannels,
        shared_state: ns_shared.SharedState,
        name,
        type,
        ip,
    ):
        self._sdk = ugot.UGOT()
        self.queue_channels = queue_channels
        self.shared_state = shared_state
        self.name = name
        self.type = type
        self.ip = ip

    def connect(self):
        # use sbbot instance to scan
        # logger.info("Scanning for devices...")
        # scan = self._sdk.scan_device()
        # for key, value in scan.items():
        #     if key == self.name:
        #         try:
        #             self._sdk.initialize(value)
        #         except Exception as e:
        #             print(e)
        #             # oh no couldn't find it
        #             logger.error(f"COULD NOT CONNECT TO {self.type} ON THE NETWORK")
        #             return
        #         logger.info(f"Succesfully connected to {self.type} at {value}")
        # logger.error(f"COULD NOT FIND {self.type} ({self.name}) ON THE NETWORK")
        with self.shared_state.peripheral_sbbot_status_lock:
            self.shared_state.peripheral_sbbot_status = (
                ns_shared.PeripheralStatus.CONNECTING
            )
        try:
            self._sdk.initialize(self.ip)
        except Exception as e:
            logger.exception(e)
            # oh no couldn't find it
            logger.error(f"COULD NOT CONNECT TO {self.type} ON THE NETWORK")
            return
        logger.info(f"Succesfully connected to {self.type} at {self.ip}")
        with self.shared_state.peripheral_sbbot_status_lock:
            self.shared_state.peripheral_sbbot_status = (
                ns_shared.PeripheralStatus.CONNECTED
            )

    def map_and_clamp(self, value, in_min, in_max, out_min, out_max):
        # 1. Constrain the input value to the input range
        value = max(in_min, min(value, in_max))

        # 2. Map the value proportionally to the output range
        ratio = (value - in_min) / (in_max - in_min)
        mapped_value = out_min + ratio * (out_max - out_min)

        # 3. Final safety clamp in case of floating-point inaccuracies
        return max(out_min, min(mapped_value, out_max))

    def calculate_mecanum_powers(self, joy_x, joy_y, joy_r, max_rpm):
        """
        Translates joystick inputs into normalized motor powers for a 4WD Mecanum robot.
        Inputs are expected to be in the range [-1.0, 1.0].
        returns tuple (front_left,front_right,back_left,back_right)
        """
        # 1. Map inputs to kinematics variables
        # (Inverting joy_y is common since most joysticks return negative when pushed forward)
        y = -joy_y
        x = joy_x
        r = joy_r

        # 2. Apply the Mecanum kinematic math
        front_left = y + x + r
        back_left = y - x + r
        front_right = y - x - r
        back_right = y + x - r

        # 3. Normalize the values so no motor exceeds 1.0 / -1.0
        # Find the largest absolute value among all 4 outputs
        max_power = max(
            abs(front_left), abs(back_left), abs(front_right), abs(back_right)
        )

        # If the largest value is greater than 1, scale everything down proportionally
        if max_power > 1.0:
            front_left /= max_power
            back_left /= max_power
            front_right /= max_power
            back_right /= max_power

        front_left = front_left * max_rpm
        front_right = front_right * max_rpm
        back_left = back_left * max_rpm
        back_right = back_right * max_rpm

        # 4. Return a dictionary of power values ready for the motors
        return (front_left, front_right, back_left, back_right)

    def SBB_AP_centralization_approaching(
        self, distance=0.15, gap=20, fwd_spd=10, turn_spd=5
    ):
        """
        Drive toward a detected AprilTag, keeping it centered in the camera frame.

        Parameters:
            distance  (float): Stop when the tag is within this many meters (default 0.15 m).
            gap       (int):   Pixel tolerance around center (320 px) before strafing (default 20 px).
            fwd_spd   (int):   Forward drive speed percentage (default 10 cm/s).
            strafe_spd(int):   Left/right correction speed percentage (default 10 cm/s).
        """
        try:
            # Get an initial reading to confirm a tag is visible before entering the loop.
            AP_info = self._sdk.get_apriltag_total_info()
            AP_x = AP_info[0][1]
            # Horizontal pixel position of the tag (0=left, 640=right)
            AP_distance = AP_info[0][6]  # Estimated distance to the tag in meters
            logger.info(f"AP_x: {AP_x}, AP_distance: {AP_distance}")

            while True:
                # Refresh tag data every iteration for responsive corrections.
                AP_info = self._sdk.get_apriltag_total_info()
                AP_x = AP_info[0][1]
                AP_distance = AP_info[0][6]

                if AP_x < 320 - gap:
                    # Tag is to the LEFT of center — strafe left to re-align.
                    # mecanum_move_xyz(x, y, z): x=strafe, y=forward, z=rotation
                    self._sdk.balance_move_turn(0, fwd_spd, 2, turn_spd)
                elif AP_x > 320 + gap:
                    # Tag is to the RIGHT of center — strafe right to re-align.
                    self._sdk.balance_move_turn(0, fwd_spd, 3, turn_spd)
                elif AP_distance > distance:
                    # Tag is centered but still too far — drive straight forward.
                    self._sdk.balance_move_speed(0, fwd_spd)
                else:
                    # Tag is centered AND within target distance — stop and exit.
                    self._sdk.balance_move_speed(0, 0)
                    logging.info("It's too close, let's stop.")
                    self._sdk.screen_display_background(6)
                    break
        except IndexError:
            logging.error("ERROR: AprilTag cannot be seen.")

    def SBB_charge_and_stop(self):
        self._sdk.balance_move_speed_times(0, 80, 100, 1)
        while True:
            self._sdk.screen_display_background(7)
            line_type = self._sdk.get_single_track_total_info()
            logging.info(f"Line type: {line_type}")

            if line_type == 1:
                while True:
                    self._sdk.screen_display_background(0)
                    line_type = self._sdk.get_single_track_total_info()
                    logging.info(f"Line type: {line_type}")

                    if line_type == 0:
                        break

                    self._sdk.balance_move_speed(0, 80)

                break

            self._sdk.balance_move_speed(0, 40)

        self._sdk.screen_display_background(6)
        self._sdk.balance_move_speed(0, 0)

    def register_villian(self):
        logging.info("Registering villian...")
        self._sdk.face_recognition_add_name("Bad Guy")
        logging.info("Villian registered.")

    def red_ball_pickup(self):
        """Detect the red ball and drive toward it; pick it up when close enough.

        uses cv2 and numpy
        """
        CAMERA_FRAME_WIDTH = 640
        CAMERA_FRAME_HEIGHT = 480

        # Horizontal dead-zone around frame centre (320 px).
        # Objects with center_x inside [LEFT_THRESHOLD, RIGHT_THRESHOLD] are treated as centred.
        LEFT_THRESHOLD = 320 - 10
        RIGHT_THRESHOLD = 320 + 10

        # Ball bounding-box width (px) at which pickup is triggered.
        # Increase if the robot is grabbing from too far away; decrease if it overshoots.
        RED_BALL_PICKUP_THRESHOLD = 200

        # Face bounding-box width (px) at the ideal throwing distance.
        # Use the Face-Annotated Camera Feed to find the right value for your arena.
        FACE_WIDTH_APPROACH_THRESHOLD = 80

        # ± tolerance on the face width target — prevents the robot oscillating around the goal.
        FACE_WIDTH_APPROACH_TOLERANCE = 10

        # --- Red ball HSV colour ranges ---
        # Red wraps around 0° in the OpenCV hue circle, so two ranges are required.
        RED_HSV_LOWER_1 = np.array([0, 70, 70])  # low-hue red  (0–10°)
        RED_HSV_UPPER_1 = np.array([10, 255, 255])
        RED_HSV_LOWER_2 = np.array([170, 70, 70])  # high-hue red (170–180°)
        RED_HSV_UPPER_2 = np.array([180, 255, 255])

        while not self.queue_channels.kill_flag.is_set():
            with self.shared_state.eng_camera_frame_lock:
                if self.shared_state.eng_camera_frame is not None:
                    img = self.shared_state.eng_camera_frame.copy()
                else:
                    logger.warning("THE THING'S NONE RIGHT NOW")
                    continue
            # Convert frame to HSV for colour-range detection
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Build a binary mask covering both red hue ranges, then combine with OR
            mask1 = cv2.inRange(hsv_img, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
            mask2 = cv2.inRange(hsv_img, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Find contours of the masked (red) regions
            # RETR_TREE retrieves the full contour hierarchy (overkill here, but harmless)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Work with the largest red region to ignore small noise blobs
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > 500:  # Minimum area threshold — filters out tiny noise specks
                    # Get the axis-aligned bounding box: top-left (x,y), width w, height h
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Annotate the debug frame with box and label
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    label = f"Center_x: {center_x} Area:{area} w:{w}"
                    cv2.putText(
                        img,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

                    if w > RED_BALL_PICKUP_THRESHOLD:
                        # Ball is large enough (close enough) — execute pickup sequence
                        self._sdk.mecanum_stop()
                        time.sleep(1)

                        # Open gripper, lower arm to ball level, close gripper, raise arm
                        # joint_control(j1_base, j2_mid, j3_tip, duration_ms)
                        # Negative j2/j3 angles tilt the arm downward toward the floor.
                        self._sdk.mechanical_clamp_release()  # Open gripper
                        self._sdk.mechanical_joint_control(
                            0, -30, -55, 1500
                        )  # Lower arm to ball
                        time.sleep(2)  # Wait for arm to reach position
                        self._sdk.mechanical_clamp_close()  # Grab the ball
                        time.sleep(1)  # Wait for gripper to close
                        self._sdk.mechanical_joint_control(
                            0, 50, 80, 1500
                        )  # Raise arm to carry position
                        state = "Search Face"  # Ball is in hand — move to next phase

                    elif center_x < LEFT_THRESHOLD:
                        # Ball is to the left of centre — strafe left to align
                        self._sdk.mecanum_move_xyz(-5, 0, 0)
                    elif center_x > RIGHT_THRESHOLD:
                        # Ball is to the right of centre — strafe right to align
                        self._sdk.mecanum_move_xyz(5, 0, 0)
                    else:
                        # Ball is centred but not yet close enough — drive forward
                        self._sdk.mecanum_move_xyz(0, 5, 0)
