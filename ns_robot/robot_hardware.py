import json
import logging
import math
import os
import shutil
import time
from math import atan
from turtle import width

import cv2
import numpy as np
from ugot import ugot
from ugot.src.http_client import upload_vision_picture

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

    def mecanum_translate(self, vx, vy, omega, max_speed=80, max_rotation_speed=280):
        self._sdk.mecanum_move_xyz(
            int(vx * max_speed), int(vy * max_speed), int(omega * max_rotation_speed)
        )

    def navigate_to_delivery_zone(
        self, target_color, MAX_SPEED=80, MAX_ROTATION_SPEED=280
    ):
        logging.info(f"Searching for the {target_color.name} delivery zone...")

        zone_located = False
        while not zone_located and not self.queue_channels.kill_flag.is_set():
            with self.shared_state.block_detection_data_lock:
                payload = self.shared_state.block_detection_data
                visible_zones = (
                    payload.get("zones", []) if isinstance(payload, dict) else []
                )

            # Find the massive zone matching our color
            target_zone = None
            for zone in visible_zones:
                if zone["color"] == target_color:
                    target_zone = zone
                    break

            if target_zone is not None:
                cx, cy = target_zone["pixel_center"]
                image_center_x = 320  # Adjust to your resolution
                pixel_error = cx - image_center_x

                # If the zone is centered enough, drive straight into it
                if abs(pixel_error) < 30:
                    logging.info("Zone centered! Driving in to deliver...")
                    self.mecanum_translate(0, 0.6, 0, MAX_SPEED, MAX_ROTATION_SPEED)

                    # Check if we have arrived (Zone bounding box takes up most of the bottom screen)
                    x, y, w, h = target_zone["pixel_bounds"]
                    if y + h > 450:  # Adjust threshold
                        logging.info("Arrived at destination zone.")
                        self._sdk.mecanum_stop()
                        zone_located = True
                else:
                    # Rotate toward the zone center
                    omega_turn = 0.15 if pixel_error > 0 else -0.15
                    self.mecanum_translate(
                        vx=0,
                        vy=0,
                        omega=omega_turn,
                        max_speed=MAX_SPEED,
                        max_rotation_speed=MAX_ROTATION_SPEED,
                    )
            else:
                # Blindly scan/rotate until the large floor graphic is sighted
                self.mecanum_translate(
                    vx=0,
                    vy=0,
                    omega=0.2,
                    max_speed=MAX_SPEED,
                    max_rotation_speed=MAX_ROTATION_SPEED,
                )

            time.sleep(0.1)

    def return_to_field(self, MAX_SPEED=80, MAX_ROTATION_SPEED=280):
        self.mecanum_translate(0, -MAX_SPEED, 0)
        time.sleep(0.5)
        self.execution_j_turn()

    def execution_j_turn(self, MAX_SPEED=40, MAX_ROTATION_SPEED=40):
        """
        Executes a high-speed 180-degree spin while translating backwards,
        creating a sweeping J-turn style maneuver.
        """
        logging.info("Executing stylish 180-degree backward sweep!")

        # 1. Fire the vectors: Negative Y (backwards) + Positive Omega (turn)
        # Adjust the vy (-0.6) and omega (1.0) balances to make the arc tighter or wider
        v_x = 0.0
        v_y = -0.6  # Move backward at 60% power
        omega = 1.0  # Spin fast

        # We pulse this combination for a brief moment
        # You will need to tune this sleep duration based on your battery level and carpet traction
        pulse_duration = 0.65  # seconds

        start_time = time.time()
        while time.time() - start_time < pulse_duration:
            self.mecanum_translate(v_x, v_y, omega, MAX_SPEED, MAX_ROTATION_SPEED)
            time.sleep(0.02)  # Fast motor update cycle

        # 2. Actively counter-brake to snap the robot out of the drift cleanly
        # (Optional, but makes it look crisp and intentional)
        self.mecanum_translate(0, 0, 0, MAX_SPEED, MAX_ROTATION_SPEED)
        self._sdk.mecanum_stop()
        logging.info("Maneuver complete.")

    def pivot_around_plough(self, deg_s, MAX_SPEED=40, MAX_ROTATION_SPEED=120):
        """
        Pivots the robot cleanly around the front plough (D cm forward from center).
        Accepts target rotation velocity directly in degrees per second (deg_s).
        """
        # Physical distance from the chassis center to the front plough tool in cm
        D = 25.0

        # 1. Calculate the clean rotation ratio for the SDK
        omega_ratio = deg_s / MAX_ROTATION_SPEED

        # 2. Kinematics Fix: To keep a front point pinned, the center must strafe (v_x).
        # Convert degrees/s to radians/s to calculate physical target velocity (cm/s)
        rad_s = math.radians(deg_s)
        target_vx_cms = rad_s * D

        # Map the physical strafe speed to the SDK's native -1.0 to 1.0 ratio
        v_x_ratio = target_vx_cms / MAX_SPEED
        v_y_ratio = 0.0  # Explicitly 0! No forward/backward velocity needed.

        # 3. Dynamic Ratio Guard
        # If the requested deg_s requires a strafe faster than MAX_SPEED can handle,
        # scale both parameters down together to protect the geometric pivot point.
        max_val = max(abs(v_x_ratio), abs(omega_ratio))
        if max_val > 1.0:
            v_x_ratio /= max_val
            omega_ratio /= max_val
            logging.warning("Requested rotation speed capped to preserve physical plough pivot limits.")

        # 4. Ship the balanced ratios directly to your translation engine
        logging.info(f"Plough Pivot -> Target: {deg_s}°/s | vx_ratio: {v_x_ratio:.3f}, omega_ratio: {omega_ratio:.3f}")
        logging.info(f"IMU: {self.get_imu_heading()}")
        self.mecanum_translate(
            v_x_ratio, v_y_ratio, omega_ratio, MAX_SPEED, MAX_ROTATION_SPEED
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
        self, distance=0.5, gap=50, fwd_spd=5, turn_spd=20, target_id=5
    ):
        """
        Drive toward the closest detected AprilTag in a path sequence, keeping it centered.
        Breaks out fully only when arriving at the final AprilTag (when only 1 tag remains).
        """
        logging.info("SBB_AP_centralization_approaching: started path following run!")

        while not self.queue_channels.kill_flag.is_set():
            if not self.shared_state.phase_state.is_running.is_set():
                break

            ui_primitives = []

            try:
                # 1. Grab raw information matrix
                raw_info = self._sdk.get_apriltag_total_info()
                if not raw_info:
                    raise IndexError

                AP_info = raw_info.copy()
                total_tags_visible = len(AP_info)

                # 2. Path Finding Strategy: Always target the closest tag
                # We sort the detection arrays by their distance element (index 6)
                AP_info.sort(key=lambda tag: float(tag[6]))

                # Target tag is now guaranteed to be at index 0
                target_tag = AP_info[0]

                AP_id = target_tag[0]
                AP_x = target_tag[1]
                AP_y = target_tag[2]
                AP_height = target_tag[3]
                AP_width = target_tag[4]
                AP_distance = float(target_tag[6])

                # --- POPULATE ACTIVE VISUALS FOR THE CURRENT TARGET ---
                half_w_norm = (AP_width / 2.0) / 640.0
                half_h_norm = (AP_height / 2.0) / 480.0
                center_x_norm = AP_x / 640.0
                center_y_norm = AP_y / 480.0

                ui_primitives.append(
                    {
                        "type": "rectangle",
                        "corners": [
                            (center_x_norm - half_w_norm, center_y_norm - half_h_norm),
                            (center_x_norm + half_w_norm, center_y_norm - half_h_norm),
                            (center_x_norm + half_w_norm, center_y_norm + half_h_norm),
                            (center_x_norm - half_w_norm, center_y_norm + half_h_norm),
                        ],
                        "color": (255, 0, 255),
                        "thickness": 2,
                    }
                )

                ui_primitives.append(
                    {
                        "type": "text",
                        "position": (0.05, 0.08),
                        "text": f"Target Dist: {round(AP_distance, 2)}m | Visible: {total_tags_visible}",
                        "color": (0, 255, 0),
                        "scale": 0.6,
                        "thickness": 1,
                    }
                )

            except IndexError:
                logging.error("No AprilTags detected in frame")
                ui_primitives.append(
                    {
                        "type": "text",
                        "position": (0.05, 0.08),
                        "text": "Searching for Path Tags...",
                        "color": (0, 0, 255),
                        "scale": 0.6,
                        "thickness": 1,
                    }
                )
                with self.shared_state.sbbot_draw_data_lock:
                    self.shared_state.sbbot_draw_data = ui_primitives
                # Search spin mode if path tracking gets completely broken
                # self._sdk.balance_move_speed(0, speed=int(0.5 * fwd_spd))
                time.sleep(0.02)
                continue

            except Exception as e:
                logging.error(f"Path processing failure: {e}")
                time.sleep(0.02)
                continue

            with self.shared_state.sbbot_draw_data_lock:
                self.shared_state.sbbot_draw_data = ui_primitives

            logging.info(
                f"Targeting closest X: {AP_x}, Dist: {AP_distance}m. ID: {AP_id}. Total tags seen: {total_tags_visible}"
            )

            # --- ARBITRATION ROUTING FOR PATH ENDPOINTS ---
            if (
                AP_distance <= distance and AP_id == target_id
                # and (AP_x > 320 - gap)
                # and (AP_x < 320 + gap)
            ):
                if total_tags_visible == 1:
                    # TRUE ENDPOINT REACHED: Close to target and no alternative tags exist
                    self._sdk.balance_stop_balancing()
                    self._sdk.screen_display_background(6)

                    with self.shared_state.sbbot_draw_data_lock:
                        self.shared_state.sbbot_draw_data = []

                    logging.info(
                        "!!! FINAL TAG IN SEQUENCE DETECTED AND REACHED: BREAKING OUT !!!"
                    )
                    break
                else:
                    # Intermediate waypoint node reached. Transition to tracking the next index tag.
                    logging.info(
                        "Waypoint tag reached. Transitioning focus to the next node..."
                    )
                    # Give the physical robot loop a quick tick window to pivot toward the next tag
                    time.sleep(0.1)
                    continue
            elif AP_distance <= distance and AP_id == target_id:
                logging.info("I COULD DASH NW BUT NO")
            # --- MOVEMENT DIRECTION CONTROLS ---
            elif AP_x < 320 - gap:
                logging.info("Correcting Left toward closest tag")
                self._sdk.balance_move_turn(0, int(1 * fwd_spd), 2, turn_spd)
            elif AP_x > 320 + gap:
                logging.info("Correcting Right toward closest tag")
                self._sdk.balance_move_turn(0, int(1 * fwd_spd), 3, turn_spd)
            elif AP_distance > distance:
                logging.info("Advancing along path chain")
                self._sdk.balance_move_speed(0, fwd_spd)

            time.sleep(0.02)

        logging.info("End of loop")
        logging.info(f"=== FUNCTION EXITED FULLY AT {time.time()} ===")

    def SBB_charge_and_stop(self):
        # charge
        self._sdk.balance_set_acceleration(10)
        self._sdk.balance_move_speed_times(0, 80, 100, 1)
        # stop
        # self.SBB_AP_centralization_approaching(
        #     distance=0.5, gap=20, fwd_spd=5, turn_spd=20, target_id=5
        # )
        # while not self.queue_channels.kill_flag.is_set():

        self._sdk.screen_display_background(6)
        self._sdk.balance_stop_balancing()

    def register_villian(self):
        logging.info("Registering villian...")
        self._sdk.face_recognition_add_name("Bad Guy")
        logging.info("Villian registered.")

    def eng_ball_centralise_and_pick(
        self,
        max_speed=20,
        strafe_speed=10,
        threshold=50,
        arm_down_distance=60,
        pick_distance=50,
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

    def eng_throw_ball(self, villain_scans=10, width_of_face=5):
        self._sdk.mecanum_stop()
        villain_data = []
        center_x_data = []
        width_data = []

        while (not self.queue_channels.kill_flag.is_set()) and len(
            villain_data
        ) < villain_scans:
            # make sure we get ten face readings. Maybe overkill, maybe shld tune.
            face_data = self._sdk.get_face_recognition_total_info()

            for _ in face_data:
                if _[0] == "villain":
                    villain_data.append(_)
                    center_x_data.append(_[1])
                    width_data.append(_[4])
                    logging.info(
                        f"Villain detected at ({_[1]},{_[2]}), {len(villain_data)}/{villain_scans}"
                    )
                    break
            if not villain_data:
                logging.warning("no villain detected bleh")
                continue

        # now that we found the guy
        # now extract face data from sdk list (and get average center x)
        # name (str): Name (or “Unknown” for unrecognized faces)
        # center_x (float): Center x-coordinate
        # center_y (float): Center y-coordinate
        # height (float): Height
        # width (float): Width
        # area (float): Area

        mean_center_x = sum(center_x_data) / len(center_x_data)
        mean_width = sum(width_data) / len(width_data)

        # 1. Calculate actual distance (Keep this if you need it for arm trajectory)
        distance = (width_of_face * ns_shared.CAMERA_FOCAL_LENGTH) / mean_width
        logging.info(f"Target is approximately {distance:.1f} cm away.")

        # 2. Calculate horizontal angular offset from center lens
        IMAGE_CENTER_X = 320  # Assuming 640 width frame
        pixel_error = mean_center_x - IMAGE_CENTER_X

        # atan returns radians; convert to degrees
        angle_to_turn_rad = math.atan(pixel_error / ns_shared.CAMERA_FOCAL_LENGTH)
        angle_to_turn = int(math.degrees(angle_to_turn_rad))

        logging.info(
            f"Target pixel offset: {pixel_error}px -> Turning {angle_to_turn} degrees to center up!"
        )

        # 3. Fire your SDK joint controls
        self._sdk.mechanical_single_joint_control(1, angle_to_turn, 500)
        time.sleep(1)
        self._sdk.mechanical_single_joint_control(2, 45, 400)
        time.sleep(0.2)
        self._sdk.mechanical_single_joint_control(3, 50, 500)
        time.sleep(0.2)
        self._sdk.mechanical_clamp_release()

    def register_face_from_file(self, name, target_jpeg_path):
        """
        Registers a face using a custom local JPEG file by copying it to the
        expected SDK execution directory before uploading.
        """
        if not os.path.exists(target_jpeg_path):
            logging.error(f"Target file not found: {target_jpeg_path}")
            return False

        names = self._sdk.VISION.face_recognition_get_all_names()
        if names and name in names:
            logging.info(f"Face [{name}] already exists.")
            return True

        self._sdk.load_models(["face_recognition"])

        # 1. Determine the SDK's expected execution directory
        sdk_dir = os.path.dirname(os.path.realpath(__file__))
        image_name = "{}.jpg".format(name)
        expected_local_path = os.path.join(sdk_dir, image_name)

        # 2. Safely copy your image into this directory with the exact expected name
        try:
            if os.path.abspath(target_jpeg_path) != os.path.abspath(
                expected_local_path
            ):
                shutil.copy2(target_jpeg_path, expected_local_path)
                logging.info(f"Staged image to script directory: {expected_local_path}")
        except Exception as e:
            logging.error(f"Failed staging image file locally: {e}")
            return False

        logging.info(f"Uploading staged image via HTTP...")

        # 3. Upload using the cleanly formatted local path
        upload_response = upload_vision_picture(
            self._sdk.http_basic_url, expected_local_path
        )

        # Clean up the staged file on your laptop immediately so it doesn't clutter your code workspace
        if os.path.exists(expected_local_path):
            os.remove(expected_local_path)

        if not upload_response or upload_response.get("code") != 0:
            logging.error("Failed to upload the image to the robot server.")
            return False

        # 4. Trigger the GRPC DB Insertion
        logging.info(f"Committing {image_name} to onboard face database...")
        response = self._sdk.VISION.face_recognition_insert_data(image_name, name)

        if response is not None:
            if response.code == 0:
                logging.info(f"Face [{name}] registered successfully!")
                return True
            else:
                logging.error(f"Robot backend error. Message: {response.msg}")
                return False

        logging.error("No response from GRPC face service.")
        return False

    def get_imu_heading(self):
        data = self._sdk.SENSOR.getIMUSensorValue()
        return data.yaw
