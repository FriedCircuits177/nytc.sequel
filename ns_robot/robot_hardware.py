import logging
import time
from math import dist

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
        self, distance=0.5, gap=20, fwd_spd=5, turn_spd=30
    ):
        """
        Drive toward a detected AprilTag, keeping it centered in the camera frame.
        """
        logging.info("SBB_AP_centralization_approaching: started the run!")

        while not self.queue_channels.kill_flag.is_set():
            if not self.shared_state.phase_state.is_running.is_set():
                break

            ui_primitives = []

            try:
                AP_info = self._sdk.get_apriltag_total_info().copy()
                AP_x = AP_info[0][1]
                AP_y = AP_info[0][2]
                AP_height = AP_info[0][3]
                AP_width = AP_info[0][4]
                AP_distance = AP_info[0][6]

                # --- 1. POPULATE ACTIVE VISUALS ---
                half_w_norm = (AP_width / 2.0) / 640.0
                half_h_norm = (AP_height / 2.0) / 480.0
                center_x_norm = AP_x / 640.0
                center_y_norm = AP_y / 480.0

                ui_primitives.append(
                    {
                        "type": "rectangle",
                        "corners": [
                            (
                                center_x_norm - half_w_norm,
                                center_y_norm - half_h_norm,
                            ),
                            (
                                center_x_norm + half_w_norm,
                                center_y_norm - half_h_norm,
                            ),
                            (
                                center_x_norm + half_w_norm,
                                center_y_norm + half_h_norm,
                            ),
                            (
                                center_x_norm - half_w_norm,
                                center_y_norm + half_h_norm,
                            ),
                        ],
                        "color": (255, 0, 255),
                        "thickness": 2,
                    }
                )

                ui_primitives.append(
                    {
                        "type": "text",
                        "position": (0.05, 0.08),
                        "text": f"({int(AP_x)}, {round(float(AP_distance), 2)}m)",
                        "color": (0, 255, 0),
                        "scale": 0.6,
                        "thickness": 1,
                    }
                )

            except IndexError:
                logging.error("No AprilTag detected bleh")
                ui_primitives.append(
                    {
                        "type": "text",
                        "position": (0.05, 0.08),
                        "text": "No AprilTag",
                        "color": (0, 0, 255),
                        "scale": 0.6,
                        "thickness": 1,
                    }
                )
                with self.shared_state.sbbot_draw_data_lock:
                    self.shared_state.sbbot_draw_data = ui_primitives
                self._sdk.balance_move_speed(0, speed=int(0.5 * fwd_spd))
                time.sleep(0.02)
                continue

            except Exception as e:
                logging.error(f"Inner processing error: {e}")
                time.sleep(0.02)
                continue

            with self.shared_state.sbbot_draw_data_lock:
                self.shared_state.sbbot_draw_data = ui_primitives

            # --- CRITICAL FIX: FORCE EXPLICIT FLOAT CASTING ---
            # --- SNAPSHOT STABILIZATION ---
            # try:
            #     current_dist = float(
            #         AP_info[0][6]
            #     )  # Lock a local snapshot copy right here
            #     target_dist = float(distance)
            # except Exception as e:
            #     logging.error(f"Snapshot tracking failed: {e}")
            #     time.sleep(0.02)
            #     continue

            # Precise diagnostic logging
            logging.info(f"{AP_x},{AP_y},{AP_distance}")

            if AP_distance <= distance:
                # self._sdk.balance_move_speed(0, 0)
                self._sdk.balance_stop_balancing()
                self._sdk.screen_display_background(6)

                with self.shared_state.sbbot_draw_data_lock:
                    self.shared_state.sbbot_draw_data = []

                logging.info("!!! EXECUTING BREAK STATEMENT NOW !!!")
                break  # This WILL kill this specific while loop.

            elif AP_x < 320 - gap:
                logging.info("Moving Left")
                self._sdk.balance_move_turn(0, fwd_spd, 2, turn_spd)
            elif AP_x > 320 + gap:
                logging.info("Moving Right")
                self._sdk.balance_move_turn(0, fwd_spd, 3, turn_spd)
            elif AP_distance > distance:
                logging.info("Moving forward")
                self._sdk.balance_move_speed(0, fwd_spd)
            time.sleep(0.02)

        logging.info("End of loop")

        # This logs the exact millisecond the function officially finishes execution
        logging.info(f"=== FUNCTION EXITED FULLY AT {time.time()} ===")

    def SBB_charge_and_stop(self):
        # charge
        self._sdk.balance_move_speed_times(0, 80, 100, 1)
        # stop

        # while not self.queue_channels.kill_flag.is_set():
        #     self._sdk.screen_display_background(7)
        #     line_type = self._sdk.get_single_track_total_info()
        #     logging.info(f"Line type: {line_type}")
        #     self._sdk.balance_move_speed(0, 40)

        #     if line_type == 1:
        #         while not self.queue_channels.kill_flag.is_set():
        #             self._sdk.screen_display_background(0)
        #             line_type = self._sdk.get_single_track_total_info()
        #             logging.info(f"Line type: {line_type}")
        #             self._sdk.balance_move_speed(0, 80)

        #             if line_type == 0:
        #                 self._sdk.balance_move_speed(0, 0)
        #                 break

        #         self._sdk.balance_move_speed(0, 0)
        #         break

        self._sdk.screen_display_background(6)
        self._sdk.balance_move_speed(0, 0)

    def register_villian(self):
        logging.info("Registering villian...")
        self._sdk.face_recognition_add_name("Bad Guy")
        logging.info("Villian registered.")

    def eng_ball_centralise_and_pick(
        self,
        max_speed=20,
        strafe_speed=10,
        threshold=50,
        arm_down_distance=10,
        pick_distance=3,
    ):
        arm_down = False
        picked = False
        # pick_confirm = False
        # original_y = -1
        while not self.queue_channels.kill_flag.is_set():
            if not self.shared_state.phase_state.is_running.is_set():
                break
            with self.shared_state.ball_detection_data_lock:
                data = self.shared_state.ball_detection_data

            if not data:
                if not picked:
                    logging.warning("no red ball data")
                    time.sleep(0.02)
                    continue
                else:
                    # excellent, the ball is picked up and out of sight
                    logging.info("weGOT IT")
                    break

            x_error = data["x_error"]
            normalized_x = data["normalized_x"]
            distance = data["distance"]
            y = data["y"]

            logging.info(f"I SAW THE BALL, I AM {x_error} off and {distance} away")

            if distance < pick_distance and arm_down:
                logger.info("PICKING")
                # original_y = y
                self._sdk.mechanical_clamp_close()
                self._sdk.mechanical_joint_control(0, 110, 90, 2000)
                self._sdk.mecanum_move_xyz(0, int(0.5 * max_speed), 0)
                picked = True
            elif distance < arm_down_distance and not arm_down:
                logger.info("ARM COMING DOWN")
                self._sdk.mecanum_move_xyz(0, int(0.5 * max_speed), 0)
                self._sdk.mechanical_clamp_release()
                self._sdk.mechanical_joint_control(0, -60, -45, 1000)
                arm_down = True
            elif x_error > (0 + threshold):
                logger.info("GO RIGHT")
                # thatmeans it's to the right
                self._sdk.mecanum_move_xyz(strafe_speed, max_speed, 0)

            elif x_error < (0 - threshold):
                logger.info("GO LEFT")
                # that means it's to the left i guess
                self._sdk.mecanum_move_xyz(-strafe_speed, max_speed, 0)

            else:
                logger.info("GO STRAIGHT")
                # within acceptable centre, so go forward
                self._sdk.mecanum_move_xyz(0, max_speed, 0)

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
                        import cv2
                        import numpy as np

    def detect_horizontal_black_line(self, frame):
        if frame is None:
            return None

        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Blur to smooth out wood grain texture noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Sobel Y-gradient: Detects horizontal changes (edges that run left-to-right)
        # cv2.CV_16S prevents clipping of negative gradients (going from light to dark floor)
        sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)

        # 4. Threshold to isolate the strongest horizontal edges
        _, thresh = cv2.threshold(abs_sobel_y, 50, 255, cv2.THRESH_BINARY)

        # 5. Morphological Close: Bridge small gaps across the line length
        kernel = np.ones(
            (3, 15), np.uint8
        )  # Wide kernel to emphasize horizontal structures
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 6. Find contours of the edges
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_line = None
        max_width = 0

        for contour in contours:
            # Get a straight bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # --- The Anti-Floorboard Filters ---
            # 1. Reject tiny noise
            if w < 50 or h < 5:
                continue

            # 2. Aspect Ratio: A horizontal line must be significantly wider than it is tall
            aspect_ratio = w / float(h)
            if aspect_ratio < 4.0:  # Adjust this if your line is thicker/closer
                continue

            # 3. Confirm it's actually dark/black
            # Sample the pixels inside the bounding box from the original gray image
            roi = gray[y : y + h, x : x + w]
            mean_brightness = np.mean(roi)
            if (
                mean_brightness > 100
            ):  # Reject if the inside is too bright (not a black line)
                continue

            # Keep the widest matching horizontal line
            if w > max_width:
                max_width = w
                # Center coordinates of the detected line
                center_x = x + (w / 2)
                center_y = y + (h / 2)
                best_line = (center_x, center_y, w, h)

        return best_line  # Returns (cx, cy, width, height) or None

    def eng_find_face_and_stop_line(self, y_threshold):
        while not self.queue_channels.kill_flag.is_set():
            with self.shared_state.eng_camera_frame_lock:
                frame = self.shared_state.eng_camera_frame
            data = self.detect_horizontal_black_line(frame)
            if not data:
                logging.warning("NO LINE DETECTED")
                time.sleep(0.02)
                continue

    def eng_throw_ball(self):
        pass
