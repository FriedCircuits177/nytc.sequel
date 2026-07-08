import json
import logging
import math
import os
import queue
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

    def new_block_sorting(
        self,
        MAX_SPEED=40,
        MAX_DIVE_SPEED=50,
        MAX_STRAFE_SPEED=40,
        MAX_ROTATION_SPEED=20,
        alignment_tolerance=40,
    ):
        logging.info("STARTING BLOCK SORTING. HERE GOES NOTHING.")
        # NORTH_HEADING = (
        #     self.get_imu_heading()
        # )  # this refers to the end with the delivery zones
        self.p4_target_heading = self.get_imu_heading()
        self.p4_current_heading = self.p4_target_heading
        plough = []
        vision_data = {}
        block_line = []
        delivery_tolerance_y = 400
        state = ns_shared.MicroState.P4_CALIBRATE_LEFT
        delivery_schedule = ns_shared.DeliverySchedule()
        delivery_schedule.add(ns_shared.BlockColour.RED, 1)
        delivery_schedule.add(ns_shared.BlockColour.RED, 2)
        delivery_schedule.add(ns_shared.BlockColour.BLUE, 2)
        self._sdk.mecanum_move_speed_times(0, 60, 30, 1)  # move onto field

        while not self.queue_channels.kill_flag.is_set():
            self._sdk.screen_display_background(0)

            if not self.shared_state.phase_state.is_running.is_set():
                self._sdk.mecanum_stop()
                logging.info("Block sorting halted: Killed by user")
                return
            # first, poll data if needed
            if state in (
                ns_shared.MicroState.P4_SCAN,
                ns_shared.MicroState.P4_ALIGN,
                ns_shared.MicroState.P4_DELIVER,
            ):
                with self.shared_state.block_detection_data_lock:
                    vision_data = self.shared_state.block_detection_data.copy()
                if not vision_data:
                    logging.warning(
                        f"Block sorting: No vision data was detected. Currently on {state}"
                    )
                    time.sleep(0.01)
                    continue
                if len(vision_data["blocks"]) > 5:
                    logging.warning(
                        f"Block sorting: Detected blocks > 5. Detected {len(vision_data['blocks'])}. Skipping this frame"
                    )
                    time.sleep(0.01)
                    continue
                block_line = sorted(
                    vision_data["blocks"], key=lambda item: item["pixel_center"][0]
                )  # sorts blocks from left to right

            match state:
                case ns_shared.MicroState.P4_CALIBRATE_LEFT:
                    logging.info("Aligning left")
                    self.bang_wall("left", 40, True)
                    logging.info("Aligning left done")
                    state = ns_shared.MicroState.P4_SCAN

                case ns_shared.MicroState.P4_CALIBRATE_RIGHT:
                    logging.info("Aligning right")
                    self.bang_wall("right", 40, True)
                    logging.info("Aligning right done")
                    state = ns_shared.MicroState.P4_SCAN

                case ns_shared.MicroState.P4_SCAN:
                    # FIX: Use sorted() to prevent block_line from becoming None
                    block_line = sorted(
                        vision_data["blocks"], key=lambda item: item["pixel_center"][0]
                    )

                    if not block_line:
                        continue

                    target_color = delivery_schedule.get_current_colour()
                    best_candidate = None
                    highest_priority = -1  # Lower values mean lower priority

                    for index, block in enumerate(block_line):
                        # Filter for current schedule color requirements
                        if block["color"] != target_color:
                            continue

                        # Determine structural priority
                        # Priority 2: Absolute edges (Index 0 or Last index) - SAFEST
                        if index == 0 or index == len(block_line) - 1:
                            current_priority = 2
                        # Priority 1: One layer inward (Index 1 or Second to last) - NEXT BEST
                        elif index == 1 or index == len(block_line) - 2:
                            current_priority = 1
                        # Priority 0: Deep interior blocks (high risk of double flanking collisions)
                        else:
                            current_priority = 0

                        # Selection/Tie-breaker logic
                        if current_priority > highest_priority:
                            highest_priority = current_priority
                            best_candidate = block
                        elif current_priority == highest_priority:
                            # If two blocks share edge priority, choose the one closer
                            # to the center of the camera frame (320px) to minimize strafing distance
                            if best_candidate:
                                current_dist = abs(block["pixel_center"][0] - 320)
                                best_dist = abs(best_candidate["pixel_center"][0] - 320)
                                if current_dist < best_dist:
                                    best_candidate = block

                    # If a suitable block matching the schedule was found, lock on and move to align
                    if best_candidate is not None:
                        # Store the chosen target in self so P4_ALIGN can track it across frames
                        self.tracked_target_block = best_candidate
                        if target_color == ns_shared.BlockColour.RED:
                            self._sdk.screen_display_background(3)
                        elif target_color == ns_shared.BlockColour.BLUE:
                            self._sdk.screen_display_background(8)
                        logging.info(
                            f"Target selected! Color: {target_color}, CX: {best_candidate['pixel_center'][0]}, Priority Tier: {highest_priority}"
                        )
                        state = ns_shared.MicroState.P4_ALIGN
                    else:
                        logging.warning(
                            f"No blocks matching color {target_color} found in the line."
                        )

                case ns_shared.MicroState.P4_ALIGN:
                    # 1. Sort the fresh frame from left to right
                    block_line = sorted(
                        vision_data["blocks"], key=lambda item: item["pixel_center"][0]
                    )

                    if not block_line:
                        # If we lose visibility entirely, stop moving and wait for a frame recovery
                        self._sdk.mecanum_stop()
                        logging.warning(
                            "P4_ALIGN: Blinded! No blocks visible. Holding position."
                        )
                        continue

                    # 2. Re-locate our tracked target block in the new frame
                    # Look for the block of the same color closest to our last target's CX coordinate
                    target_color = delivery_schedule.get_current_colour()
                    last_known_cx = self.tracked_target_block["pixel_center"][0]

                    current_target = None
                    best_match_distance = float("inf")

                    for block in block_line:
                        if block["color"] != target_color:
                            continue

                        dist = abs(block["pixel_center"][0] - last_known_cx)
                        if dist < best_match_distance:
                            best_match_distance = dist
                            current_target = block

                    # Fallback if our specific target block disappeared completely
                    if current_target is None:
                        logging.warning(
                            "P4_ALIGN: Target lost from frame! Dropping back to SCAN."
                        )
                        self._sdk.mecanum_stop()
                        state = ns_shared.MicroState.P4_SCAN
                        continue

                    # Update our tracker with the latest confirmed coordinates
                    self.tracked_target_block = current_target
                    cx = current_target["pixel_center"][0]

                    # 3. Calculate alignment errors
                    FRAME_CENTER_X = 320
                    pixel_error = cx - FRAME_CENTER_X

                    # Compute heading correction relative to your North target
                    # (Assuming get_imu_heading() returns degrees)

                    rotation_error_deg = self.p4_target_heading - self.get_imu_heading()

                    # 4. Check if we are aligned within tolerance
                    # Convert alignment_tolerance from pixels if needed, or check absolute pixel error
                    if abs(pixel_error) <= alignment_tolerance:
                        # Success! Stop sideways movement and prepare to dive down the tunnel
                        self._sdk.mecanum_stop()
                        logging.info(
                            f"P4_ALIGN: Alignment achieved! Error: {pixel_error}px. Advancing to DASH."
                        )
                        state = ns_shared.MicroState.P4_DASH
                    else:
                        # 5. Execute stationary strafe alignment
                        # Normalize pixel error to a clean -1.0 to 1.0 range
                        # -320px error = -1.0 (Full Left Strafe) | +320px error = 1.0 (Full Right Strafe)
                        normalized_strafe = pixel_error / 320.0

                        # Apply Proportional scaling logic for the commands before handing over
                        KP_STRAFE = (
                            1.5  # Controls how aggressively it ramps up to -1.0/1.0
                        )
                        KP_ROTATION = (
                            2.0  # Translates degrees error to a target velocity (deg/s)
                        )

                        # Calculate commands, clamping the normalized strafe between -1.0 and 1.0
                        strafe_cmd = max(-1.0, min(1.0, normalized_strafe * KP_STRAFE))
                        rotation_cmd_degs = rotation_error_deg * KP_ROTATION

                        # Keep Y=0! Do not nose-dive forward until horizontal alignment is complete.
                        # x: -1.0 to 1.0 | y: -1.0 to 1.0 | r: deg/s
                        self.mecanum_translate(
                            strafe_cmd,
                            0.0,
                            rotation_cmd_degs,
                            MAX_SPEED,
                            MAX_ROTATION_SPEED,
                            MAX_STRAFE_SPEED,
                        )
                case ns_shared.MicroState.P4_DASH:
                    # dash and collect in plough, maintaining heading to whatever is the current direction
                    # check constantly to see if the block entered the plough, this part can be done later. for now assume it is a succesful catch
                    # Now, make a decision based on delivery schedule etc. Turn around plough and collect another, or deliver? Change state accordingly.

                    # 1. Gather current frame details for wall clearing metrics
                    # Note: We don't sort here because we only care about the absolute count of any color blocks remaining in view
                    blocks_in_view = vision_data["blocks"]

                    # 2. Check for the Exit Condition (Have we passed the wall?)
                    if len(blocks_in_view) == 0:
                        # -------------------------------------------------------------
                        # ASSUMPTION LAND:
                        # We have physically driven past the horizontal wall line.
                        # Since the target block was aligned directly in our path,
                        # we assume that it has now successfully been collected into our plough.
                        # -------------------------------------------------------------
                        target_color = delivery_schedule.get_current_colour()
                        plough.append(target_color)
                        logging.info(
                            f"P4_DASH: Wall cleared! Successfully collected {target_color}. Plough content: {plough}"
                        )

                        # Stop the chassis immediately to avoid smashing into the delivery wall
                        self._sdk.mecanum_stop()

                        # now logic check. what to do next?
                        if delivery_schedule.get_current_quantity() == len(plough):
                            logging.info("delivering!")
                            if target_color == ns_shared.BlockColour.RED:
                                self.bang_wall(
                                    "left", back_to_centre=False
                                )  # red, bang left
                            else:
                                self.bang_wall(
                                    "right", back_to_centre=False
                                )  # blue, bang right
                            state = ns_shared.MicroState.P4_DELIVER
                        elif delivery_schedule.get_current_quantity() > len(plough):
                            logging.info("looping for another one")
                            state = ns_shared.MicroState.P4_TURN_AROUND_PLOUGH
                        time.sleep(0.02)
                        continue

                    # 3. Heading Stabilization Loop (IMU-driven straight tracking)
                    # We look forward along self.p4_target_heading (North). X=0 means no strafing allowed during the sprint.
                    rotation_error_deg = self.p4_target_heading - self.get_imu_heading()
                    KP_ROTATION_DASH = 2.5  # Tight constraint to keep chassis parallel to the 1m narrow walls
                    rotation_cmd_degs = rotation_error_deg * KP_ROTATION_DASH

                    # 4. Proportional Velocity & Slew Ramping Math (Anti-Jerk Logic)
                    # Initialize target forward speed to full throttle
                    target_y_power = 1.0

                    # Proportional Slowdown Window: As we get extremely close to the block,
                    # use its distance_z (or distance tracking approximation) to damp final approach speeds
                    # We search for our specific tracking target to find its depth
                    last_known_cx = (
                        self.tracked_target_block["pixel_center"][0]
                        if hasattr(self, "tracked_target_block")
                        else 320
                    )
                    tracked_block_now = min(
                        blocks_in_view,
                        key=lambda b: abs(b["pixel_center"][0] - last_known_cx),
                        default=None,
                    )

                    if tracked_block_now and "distance_z" in tracked_block_now:
                        z_dist = tracked_block_now["distance_z"]
                        # Adjust limits depending on your exact distance units (e.g., mm vs cm vs normalized values)
                        SLOWDOWN_THRESHOLD_Z = 150.0
                        MIN_SECURE_DASH_SPEED = 0.35  # The floor velocity so the robot never fully stalls out before hitting the block

                        if z_dist < SLOWDOWN_THRESHOLD_Z:
                            KP_FORWARD_DAMP = 1.0 / SLOWDOWN_THRESHOLD_Z
                            target_y_power = max(
                                MIN_SECURE_DASH_SPEED, z_dist * KP_FORWARD_DAMP
                            )

                    # 5. Slew Acceleration Limiter (Prevents snapping start-line jerks)
                    # We track the last sent Y command using a persistent attribute to step speed incrementally
                    if not hasattr(self, "_last_dash_y"):
                        self._last_dash_y = 0.0

                    MAX_FORWARD_JUMP_PER_STEP = (
                        0.15  # Controls transition ramp profile from standstill to MAX
                    )
                    y_delta = target_y_power - self._last_dash_y

                    if y_delta > MAX_FORWARD_JUMP_PER_STEP:
                        actual_y_cmd = self._last_dash_y + MAX_FORWARD_JUMP_PER_STEP
                    else:
                        actual_y_cmd = target_y_power

                    self._last_dash_y = actual_y_cmd

                    # 6. Dispatch Translation Execution Vector
                    # x: 0.0 (Strictly Straight Locked) | y: Smoothly Accelerated | r: IMU Guided
                    self.mecanum_translate(
                        0.0,
                        actual_y_cmd,
                        rotation_cmd_degs,
                        MAX_DIVE_SPEED,  # Passes your newly requested velocity clamp
                        MAX_ROTATION_SPEED,
                        MAX_STRAFE_SPEED,
                    )
                case ns_shared.MicroState.P4_DELIVER:
                    # deliver
                    edge_y = -1
                    for zone in vision_data["zones"]:
                        if zone["color"] != target_color:
                            edge_y = zone["bottom_edge_y"]

                    if edge_y >= delivery_tolerance_y:
                        # we are within tolerance.
                        self._sdk.mecanum_stop()
                        self._sdk.mecanum_move_speed_times(0, 30, 20, 1)
                        time.sleep(2)
                        self._sdk.mecanum_move_speed_times(0, 60, 20, 1)
                        time.sleep(1)
                        state = ns_shared.MicroState.P4_TURN_AROUND_PLOUGH
                        logging.log("DELIVERED")
                        continue

                    self._sdk.mecanum_move_xyz(0, 20, 0)

                case ns_shared.MicroState.P4_TURN_AROUND_PLOUGH:
                    # use imu and self.pivot_around_plough(), and turn around to snap directly to the opposite direction. After this back to scan.
                    pass
                case ns_shared.MicroState.P4_TURN_AROUND:
                    # this is for post-delivery, where we can turn around as compact as possible without worrying about plough contents. After this back to scan.
                    pass

            time.sleep(0.02)

        return

    def bang_wall(self, direction, speed_multiplier=40, back_to_centre=True):
        """
        Dictionary, left or right
        """
        tuneable_value = 3
        value = 0
        count = 0
        velocity = 0
        if direction == "left":
            value = -1
        elif direction == "right":
            value = 1

        logging.info(f"{value}")

        while not self.queue_channels.kill_flag.is_set():
            self._sdk.mecanum_move_xyz(
                int(value * speed_multiplier),
                0,
                0,
            )
            count += 1

            velocity = self.get_imu_gyro_x()
            logging.info(f"{velocity}")

            if count >= 10:
                if velocity < tuneable_value:
                    self._sdk.mecanum_stop()
                    time.sleep(0.25)
                    self.p4_target_heading = self.get_imu_heading()
                    time.sleep(0.1)
                    break
            time.sleep(0.05)

        logging.info("Centerlising")

        if back_to_centre:
            # self._sdk.mecanum_translate_speed_times(180 * value, 40, 50, 1)
            # self._sdk.mecanum_move_xyz(36, 0, 0)
            # sleep(5)

            self._sdk.mecanum_move_xyz((40 * value * -1), 0, 0)
            time.sleep(1)
            self._sdk.mecanum_stop()
            logging.info("Centerlising done")
            return

    def mecanum_translate(
        self, vx, vy, omega, max_speed=80, max_rotation_speed=280, max_strafe_speed=80
    ):
        self._sdk.mecanum_move_xyz(
            int(vx * max_strafe_speed),
            int(vy * max_speed),
            int(omega * max_rotation_speed),
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
        D = 19.0

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
            logging.warning(
                "Requested rotation speed capped to preserve physical plough pivot limits."
            )

        # 4. Ship the balanced ratios directly to your translation engine
        logging.info(
            f"Plough Pivot -> Target: {deg_s}°/s | vx_ratio: {v_x_ratio:.3f}, omega_ratio: {omega_ratio:.3f}"
        )
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
        time.sleep(3)
        self._sdk.balance_move_speed_times(0, 40, 50, 1)
        self._sdk.screen_display_background(6)
        time.sleep(3)
        self._sdk.balance_stop_balancing()

    # def register_villian(self):
    #     logging.info("Registering villian...")
    #     self._sdk.face_recognition_add_name("Bad Guy")
    #     logging.info("Villian registered.")

    def eng_ball_centralise_and_pick(
        self,
        max_speed=10,
        strafe_speed=5,
        threshold=70,
        arm_down_distance=20,
        pick_distance=15,
    ):
        self._sdk.screen_clear()
        self._sdk.mechanical_clamp_release()
        self._sdk.mechanical_joint_control(0, 0, 0, 1000)
        self._sdk.mecanum_move_speed_times(0, 70, 70, 1)
        time.sleep(3)
        pick_distance_attempt = 0
        stop_attempt = 0
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
                # else:
                #     # excellent, the ball is picked up and out of sight
                #     logging.info("weGOT IT")
                #     break
            self._sdk.screen_display_background(3)

            x_error = data["x_error"]
            normalized_x = data["normalized_x"]
            distance = data["distance"]
            y = data["y"]

            logging.info(f"I SAW THE BALL, I AM {x_error} off and {distance} away")

            if distance < pick_distance:
                if pick_distance_attempt < 2:
                    pick_distance_attempt += 1
                    time.sleep(0.05)
                    continue
                logger.info("PICKING")
                self._sdk.screen_display_background(6)
                self._sdk.mechanical_clamp_release()
                # original_y = y
                self._sdk.mecanum_stop()
                logger.info(f"{distance}")
                self._sdk.mecanum_move_speed_times(0, 20, int((distance * 0.5 - 4)), 1)
                time.sleep(1)
                self._sdk.mecanum_stop()
                time.sleep(0.5)
                self._sdk.mechanical_joint_control(0, -30, -70, 750)
                time.sleep(0.75)
                self._sdk.mechanical_joint_control(0, -30, -65, 750)
                time.sleep(0.75)
                # self._sdk.mechanical_joint_control(0, -30, -60, 1000)
                # time.sleep(1)
                # self._sdk.mechanical_joint_control(0, -30, -65, 1000)
                # time.sleep(1)
                self._sdk.mechanical_clamp_close()
                time.sleep(3)
                # self._sdk.mecanum_move_xyz(0, int(0.5 * max_speed), 0)
                picked = True
                self._sdk.screen_display_background(0)
                break
            # elif distance < arm_down_distance and not arm_down:
            #     logger.info("ARM COMING DOWN")
            #     self._sdk.mecanum_move_xyz(0, int(0.5 * max_speed), 0)
            #     self._sdk.mechanical_clamp_release()
            #     self._sdk.mechanical_joint_control(0, 0, -60, 1000)
            #     arm_down = True

            elif x_error > (0 + threshold):
                logger.info("GO RIGHT")
                # thatmeans it's to the right
                self._sdk.mecanum_move_xyz(strafe_speed, int(max_speed), 0)

            elif x_error < (0 - threshold):
                logger.info("GO LEFT")
                # that means it's to the left i guess
                self._sdk.mecanum_move_xyz(-strafe_speed, int(max_speed), 0)

            else:
                logger.info("GO STRAIGHT")
                # within acceptable centre, so go forward
                self._sdk.mecanum_move_xyz(0, max_speed, 0)

    # def detect_horizontal_black_line(self, frame):
    #     if frame is None:
    #         return None

    #     # 1. Convert to grayscale
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # 2. Blur to smooth out wood grain texture noise
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #     # 3. Sobel Y-gradient: Detects horizontal changes (edges that run left-to-right)
    #     # cv2.CV_16S prevents clipping of negative gradients (going from light to dark floor)
    #     sobel_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    #     abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    #     # 4. Threshold to isolate the strongest horizontal edges
    #     _, thresh = cv2.threshold(abs_sobel_y, 50, 255, cv2.THRESH_BINARY)

    #     # 5. Morphological Close: Bridge small gaps across the line length
    #     kernel = np.ones(
    #         (3, 15), np.uint8
    #     )  # Wide kernel to emphasize horizontal structures
    #     closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    #     # 6. Find contours of the edges
    #     contours, _ = cv2.findContours(
    #         closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #     )

    #     best_line = None
    #     max_width = 0

    #     for contour in contours:
    #         # Get a straight bounding rectangle
    #         x, y, w, h = cv2.boundingRect(contour)

    #         # --- The Anti-Floorboard Filters ---
    #         # 1. Reject tiny noise
    #         if w < 50 or h < 5:
    #             continue

    #         # 2. Aspect Ratio: A horizontal line must be significantly wider than it is tall
    #         aspect_ratio = w / float(h)
    #         if aspect_ratio < 4.0:  # Adjust this if your line is thicker/closer
    #             continue

    #         # 3. Confirm it's actually dark/black
    #         # Sample the pixels inside the bounding box from the original gray image
    #         roi = gray[y : y + h, x : x + w]
    #         mean_brightness = np.mean(roi)
    #         if (
    #             mean_brightness > 100
    #         ):  # Reject if the inside is too bright (not a black line)
    #             continue

    #         # Keep the widest matching horizontal line
    #         if w > max_width:
    #             max_width = w
    #             # Center coordinates of the detected line
    #             center_x = x + (w / 2)
    #             center_y = y + (h / 2)
    #             best_line = (center_x, center_y, w, h)

    #     return best_line  # Returns (cx, cy, width, height) or None

    # def eng_find_face_and_stop_line(self, y_threshold):
    #     while not self.queue_channels.kill_flag.is_set():
    #         with self.shared_state.eng_camera_frame_lock:
    #             frame = self.shared_state.eng_camera_frame
    #         data = self.detect_horizontal_black_line(frame)
    #         if not data:
    #             logging.warning("NO LINE DETECTED")
    #             time.sleep(0.02)
    #             continue

    def eng_throw_ball(self, villain_scans=1, width_of_face=5):
        self._sdk.mechanical_joint_control(0, 90, 60, 1000)

        self._sdk.mecanum_stop()
        villain_data = []
        center_x_data = []
        width_data = []

        force_mult_k = 10

        while len(villain_data) < villain_scans:
            # make sure we get five face readings. Maybe overkill, maybe shld tune.
            if self.queue_channels.kill_flag.is_set():
                return
            face_data = self._sdk.get_face_recognition_total_info()

            for _ in face_data:
                if _[0].startswith("villain"):
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

        # Safety check to prevent division by zero if data collection failed
        # if not center_x_data or not width_data:
        #     logging.warning("Aborting: Collected lists are empty.")
        #     return

        mean_center_x = sum(center_x_data) / len(center_x_data)
        mean_width = sum(width_data) / len(width_data)

        # Constants for tracking geometry
        IMAGE_CENTER_X = 320  # Assuming 640 width frame
        DEPTH_Y = 40.0  # Constant distance from robot to row of faces in cm

        # 1. Calculate horizontal pixel error from the center lens
        pixel_error = mean_center_x - IMAGE_CENTER_X

        # 2. Convert pixel error to physical centimeters using the scaling ratio
        # (pixels) * (cm per pixel) = cm offset
        offset_x_cm = pixel_error * (width_of_face / mean_width)

        # 3. Calculate true absolute distance (Hypotenuse of the triangle)
        # Using Pythagorean theorem instead of focal length
        actual_distance = math.sqrt(offset_x_cm**2 + DEPTH_Y**2)
        logging.info(
            f"Target horizontal offset: {offset_x_cm:.1f} cm. Straight line distance: {actual_distance:.1f} cm."
        )

        # 4. Calculate the turn angle using right-triangle trigonometry
        # atan(opposite / adjacent) -> atan(offset_x_cm / DEPTH_Y)
        angle_to_turn_rad = math.atan(offset_x_cm / DEPTH_Y)
        angle_to_turn = -int(math.degrees(angle_to_turn_rad))

        logging.info(
            f"Target pixel offset: {pixel_error:.1f}px -> Turning {angle_to_turn} degrees to center up!"
        )

        # 5. Fire your SDK joint controls
        self._sdk.mechanical_joint_control(angle_to_turn, 90, 60, 1000)
        time.sleep(1)
        self._sdk.mechanical_joint_control(
            int(angle_to_turn),
            -5,
            -10,
            200,
        )
        time.sleep(0.1)
        self._sdk.mechanical_clamp_release()
        time.sleep(1)

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

    def get_imu_gyro_x(self):
        data = self._sdk.SENSOR.getIMUSensorValue()
        return data.gyro_x
