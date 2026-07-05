import logging
import queue
import time

from numpy.testing import print_assert_equal

import ns_robot
import ns_shared
from ns_shared.exceptions import PhaseAbortedException

logger = logging.getLogger(__name__)

import importlib

import cv2
import numpy as np
from ugot import ugot


class RobotController:
    def __init__(
        self,
        sbbot: ns_robot.RobotHardware,
        engbot: ns_robot.RobotHardware,
        QueueChannels: ns_shared.QueueChannels,
        SharedState: ns_shared.SharedState,
    ):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.engbot = engbot
        self.sbbot = sbbot

        self.async_setup_engbot()
        self.async_setup_sbbot()

    # def _check(self):
    #     """Checks if the frontend requested a stop.
    #     If true, throws the exception to instantly drop out of the phase."""
    #     if (
    #         self.queue_channels.kill_flag.is_set()
    #         or self.queue_channels.force_stop_phase_flag.is_set()
    #     ):
    #         self.queue_channels.force_stop_phase_flag.clear()
    #         raise PhaseAbortedException("Phase execution interrupted by user request.")

    # def _sleep(self, seconds: float):
    #     """A cancel-aware sleep helper so time.sleep doesn't lock up cancellations."""
    #     start_time = time.time()
    #     while time.time() - start_time < seconds:
    #         self._check()
    #         time.sleep(0.05)

    def async_setup_engbot(self):
        _ = ns_shared.construct_thread(self.setup_engbot)
        _.start()

    def setup_engbot(self):
        try:
            logger.info("Attempting to connect ENGBot...")
            self.engbot.connect()
            logger.info("Loading ENGBot models...")
            self.engbot._sdk.load_models(
                [
                    "color_recognition",
                    "word_recognition",
                    "line_recognition",
                    "face_recognition",
                    "apriltag_qrcode",
                ]
            )
            self.engbot.register_face_from_file("villain", ns_shared.VILLAIN_JPEG_PATH)
            self.engbot._sdk.mechanical_joint_control(0, 90, 0, 1000)
        except Exception as e:
            logger.error(f"ENGBot initialization failed: {e}. Running offline.")

    def async_setup_sbbot(self):
        _ = ns_shared.construct_thread(self.setup_sbbot)
        _.start()

    def setup_sbbot(self):
        try:
            logger.info("Attempting to connect SBBot...")
            self.sbbot.connect()
            self.sbbot._sdk.balance_start_balancing()
            # self.sbbot._sdk.balance_set_acceleration(0.5)
            self.sbbot._sdk.load_models(["apriltag_qrcode"])
        except Exception as e:
            logger.error(f"SBBot initialization failed: {e}. Running offline.")

    def mainloop(self):
        """Listens to the process_manager and executes ordered autonomous sequences."""
        while not self.queue_channels.kill_flag.is_set():
            if not self.shared_state.phase_state.is_running.is_set():
                self.sbbot._sdk.balance_start_balancing()
                time.sleep(0.02)
                continue

            try:
                with self.shared_state.phase_state.lock:
                    current_idx = self.shared_state.phase_state.current_phase_index
                    if current_idx is None or current_idx >= len(
                        self.shared_state.phase_state.phase_queue
                    ):
                        self.shared_state.phase_state.is_running.clear()
                        continue
                    current_phase = self.shared_state.phase_state.phase_queue[
                        current_idx
                    ]

                match current_phase:
                    case ns_shared.Phase.Phase1:
                        self.phase1()
                    case ns_shared.Phase.Phase2:
                        self.phase2()
                    case ns_shared.Phase.Phase2A:
                        self.phase2a()
                    case ns_shared.Phase.Phase3:
                        self.phase3()
                    case ns_shared.Phase.Phase4:
                        self.phase4()
                    case ns_shared.Phase.Phase4A:
                        self.phase4a()
                    case _:
                        logger.error(
                            f"The phase {current_phase} has no function tied to it!"
                        )
                        self.advance_phase()

            except PhaseAbortedException as e:
                logger.warning(f"Abort Signal Caught: {e}")
                self.kill_bots()

                with self.shared_state.phase_state.lock:
                    self.shared_state.phase_state.is_running.clear()
                    self.shared_state.phase_state.current_phase_index = None
                continue

            except Exception as e:
                logger.exception(f"Unexpected crash inside Main Loop: {e}")
                self.kill_bots()

        self.kill_bots()

    def kill_bots(self):
        """Safely stops all hardware components if they are connected."""
        try:
            self.sbbot._sdk.balance_stop_balancing()
        except Exception as e:
            if (
                getattr(self.shared_state, "peripheral_sbbot_status", None)
                == ns_shared.PeripheralStatus.CONNECTED
            ):
                logger.exception(e)
                logger.warning("FAILED TO STOP SBBOT, WARNING!")
        try:
            self.engbot._sdk.mecanum_stop()
        except Exception as e:
            if (
                getattr(self.shared_state, "peripheral_engbot_status", None)
                == ns_shared.PeripheralStatus.CONNECTED
            ):
                logger.exception(e)
                logger.warning("FAILED TO STOP ENGBOT, WARNING!")

    # --- REVERTED ORIGINAL PHASE METHODS ---

    def phase1(self):
        """sbb moves to april tag"""
        self.sbbot.SBB_AP_centralization_approaching(
            distance=0.4, gap=20, fwd_spd=20, turn_spd=10
        )
        self.sbbot.SBB_charge_and_stop()
        self.advance_phase()
        # success so we return True
        return True

    def phase2(self):
        # Call red ball pickup code
        # self.engbot.red_ball_pickup()
        # # find villain
        # self.engbot.search_villain()
        # # align and throw at pos 1,2,3
        # self.engbot.beat_up()
        self.queue_channels.ball_detection_active_flag.set()
        self.engbot.eng_ball_centralise_and_pick()
        self.queue_channels.ball_detection_active_flag.clear()
        self.engbot.eng_throw_ball()

        # temp
        self.engbot._sdk.mecanum_stop()

        logger.info("P2 done")
        self.advance_phase()

    def phase2a(self):
        # call opcontrol portion
        self.opcontrol()
        self.advance_phase()

    def phase3(self):
        self.opcontrol_pose()
        self.advance_phase()

    def phase4(self, MAX_SPEED=40, MAX_ROTATION_SPEED=40):
        self.queue_channels.block_detection_active_flag.set()

        CAMERA_HEIGHT = 480
        CAPTURE_THRESHOLD_Y = CAMERA_HEIGHT - 50  # Bottom 50px capture zone
        SEARCH_WINDOW = 70  # Frame-to-frame pixel tracking radius

        # PID Tuning Parameter for Vision Tracking Loop (Proportional Gain)
        # Adjust this value up if the robot tracks too sluggishly, down if it oscillates
        Kp = 0.0015

        delivery_schedule = [
            {"color": ns_shared.BlockColour.BLUE, "qty": 2},
            {"color": ns_shared.BlockColour.RED, "qty": 2},
            {"color": ns_shared.BlockColour.RED, "qty": 1},
        ]

        for trip in delivery_schedule:
            target_color = trip["color"]
            target_qty = trip["qty"]
            plough_inventory = []

            logging.info(
                f"Starting trip: Collecting {target_qty} {target_color.name} blocks."
            )

            # --- SUB-STATE 1: COLLECTION LOOP ---
            locked_target = None  # Holds active target dictionary

            while (
                len(plough_inventory) < target_qty
                and not self.queue_channels.kill_flag.is_set()
            ):
                # 1. Thread-safe retrieval of detection payload
                with self.shared_state.block_detection_data_lock:
                    detection_payload = self.shared_state.block_detection_data

                if isinstance(detection_payload, dict):
                    visible_blocks = detection_payload.get("blocks", [])
                    visible_zones = detection_payload.get("zones", [])
                else:
                    visible_blocks = []
                    visible_zones = []

                # 2. Maintain Target Lock or Acquire New Target
                tracked_block = None
                if locked_target is not None:
                    for block in visible_blocks:
                        if block["color"] == target_color:
                            dist = np.hypot(
                                block["pixel_center"][0]
                                - locked_target["pixel_center"][0],
                                block["pixel_center"][1]
                                - locked_target["pixel_center"][1],
                            )
                            if dist < SEARCH_WINDOW:
                                tracked_block = block
                                break

                if tracked_block is None:
                    valid_blocks = [
                        b for b in visible_blocks if b["color"] == target_color
                    ]
                    if valid_blocks:
                        tracked_block = min(valid_blocks, key=lambda b: b["distance_z"])
                        logging.info("New target block locked via greedy depth search.")

                locked_target = tracked_block
                obstacle_x = None

                # 3. Navigation Decision Engine
                if locked_target is not None:
                    cx, cy = locked_target["pixel_center"]

                    # Check if path is blocked by an incorrect color block closer than the target
                    path_is_blocked = False
                    for block in visible_blocks:
                        if (
                            block["color"] != target_color
                            and block["pixel_center"][1] > cy
                        ):
                            if (
                                150 < block["pixel_center"][0] < 490
                            ):  # Corridor pixel boundary
                                path_is_blocked = True
                                obstacle_x = block["pixel_center"][0]
                                break

                    if path_is_blocked and obstacle_x is not None:
                        logging.warning("Path blocked! Strafing around obstacle...")
                        strafe_direction = -1.0 if obstacle_x > 320 else 1.0
                        self.engbot.mecanum_translate(
                            strafe_direction, 0, 0, MAX_SPEED, MAX_ROTATION_SPEED
                        )
                        time.sleep(0.3)  # Execute sidestep pulse
                        continue

                    # Check if block has reached collection threshold (Bottom 50px)
                    if cy >= CAPTURE_THRESHOLD_Y:
                        logging.info(
                            "Block reached bottom threshold. Engaging intake plunge."
                        )

                        # Drive forward blindly to firmly capture block into the mechanism
                        self.engbot.mecanum_translate(
                            0, 1.0, 0, MAX_SPEED, MAX_ROTATION_SPEED
                        )
                        time.sleep(0.4)
                        self.engbot._sdk.mecanum_stop()

                        # Commit to internal inventory tracking
                        plough_inventory.append(target_color)
                        locked_target = None  # Wipe target lock for next acquisition
                        time.sleep(0.5)  # Let video pipeline latency clear
                        continue

                    # --- APPROACH PROFILE (Proportional Vision Tracking) ---
                    image_center_x = 320
                    pixel_error = cx - image_center_x

                    # Compute smooth turn using a P-control calculation
                    # Cap the maximum value to avoid violent rotational snaps
                    omega_turn = np.clip(pixel_error * Kp, -0.3, 0.3)

                    # Drive forward while constantly tracking the target heading
                    self.engbot.mecanum_translate(
                        0, 0.5, omega_turn, MAX_SPEED, MAX_ROTATION_SPEED
                    )

                else:
                    # No target visible on screen
                    if len(plough_inventory) > 0:
                        # CRITICAL: We have blocks in the plough! Rotate safely around the front bumper
                        logging.info(
                            "Target lost. Pivoting around plough to look for blocks..."
                        )
                        self.engbot.pivot_around_plough(
                            0.2, MAX_SPEED, MAX_ROTATION_SPEED
                        )
                    else:
                        # Plough is completely empty. We can spin faster on center axis safely
                        logging.info(
                            "Target lost. Spinning on center axis to look for blocks..."
                        )
                        self.engbot.mecanum_translate(
                            0, 0, 0.2, MAX_SPEED, MAX_ROTATION_SPEED
                        )

                time.sleep(0.05)  # Match 20Hz cycle rate

            # --- SUB-STATE 2: DELIVERY EXECUTIVE ---
            logging.info(
                f"Plough limit reached ({len(plough_inventory)} blocks). Proceeding to delivery zone."
            )
            self.engbot.navigate_to_delivery_zone(
                target_color, MAX_SPEED, MAX_ROTATION_SPEED
            )
            self.engbot.return_to_field()

    def phase4a(self):
        # call opcontrol portion
        self.opcontrol()
        self.advance_phase()

    def you_can_crush_it_as_dry_as_a_bone(self):
        logging.info("you can kiss it you can break all the rules")

    def opcontrol_pose(self):
        logging.info("opcontrol pose started")
        self.queue_channels.pose_recog_active_flag.set()
        self.max_speed = 40  # cm/s
        self.max_rotation_speed = 120

        # Ensure the queue starts clean
        while not self.queue_channels.pose_drive.empty():
            try:
                self.queue_channels.pose_drive.get_nowait()
            except Exception:
                break

        while not self.queue_channels.kill_flag.is_set():
            if not self.shared_state.phase_state.is_running.is_set():
                break

            try:
                # BLOCK HERE until MediaPipe produces a new calculated data payload
                # Use a small timeout so the loop can cleanly exit if the kill flag gets set
                x_val, y_val, r_val = self.queue_channels.pose_drive.get(timeout=0.1)
            except queue.Empty:
                continue

            print(f"{x_val},{y_val},{r_val}")

            # Convert float vectors into native target SDK integers
            x_movement = int(
                self.engbot.map_and_clamp(x_val, -1, 1, -self.max_speed, self.max_speed)
            )
            y_movement = int(
                self.engbot.map_and_clamp(y_val, -1, 1, -self.max_speed, self.max_speed)
            )
            r_movement = int(
                self.engbot.map_and_clamp(
                    r_val, -1, 1, -self.max_rotation_speed, self.max_rotation_speed
                )
            )

            self.engbot._sdk.mecanum_move_xyz(x_movement, y_movement, r_movement)

        self.engbot._sdk.mecanum_stop()
        self.queue_channels.pose_recog_active_flag.clear()

    def opcontrol(self, joystick_control=True):
        """fall back option using mecanum_move_xyz"""
        if joystick_control:
            self.queue_channels.vibrate_flag.set()
        self.zoom = False
        while not self.queue_channels.kill_flag.is_set():
            # print("OPCONTROL IS ALIVE")
            # self._check()  # Keeps your local manual cancellation capability active inside opcontrol loops
            self.max_speed = 80  # cm/s dumb sdk lol
            self.max_rotation_speed = 280
            self.max_zoom_rpm = 300
            with self.shared_state.drive_command_lock:
                buttons = self.shared_state.controller_buttons
                x_movement = self.shared_state.drive_x
                y_movement = self.shared_state.drive_y
                r_movement = self.shared_state.drive_r
                r2 = self.shared_state.drive_r2

            if r2 > 0.4 and not self.zoom:
                self.engbot._sdk.mecanum_motor_control(
                    self.max_zoom_rpm,
                    self.max_zoom_rpm,
                    self.max_zoom_rpm,
                    self.max_zoom_rpm,
                )
                self.zoom = True
                continue
            elif r2 > 0.4 and self.zoom:
                time.sleep(0.02)
                continue
            elif r2 < 0.4 and self.zoom:
                self.zoom = False

            if buttons["cross"]:
                break
            x_movement = int(
                self.engbot.map_and_clamp(
                    x_movement, -1, 1, -self.max_speed, self.max_speed
                )
            )
            y_movement = int(
                self.engbot.map_and_clamp(
                    y_movement, -1, 1, -self.max_speed, self.max_speed
                )
            )
            r_movement = int(
                self.engbot.map_and_clamp(
                    r_movement, -1, 1, -self.max_rotation_speed, self.max_rotation_speed
                )
            )

            self.engbot._sdk.mecanum_move_xyz(x_movement, y_movement, r_movement)
            time.sleep(0.02)
        self.engbot._sdk.mecanum_stop()

    def advance_phase(self):
        with self.shared_state.phase_state.lock:
            if self.shared_state.phase_state.current_phase_index is None:
                self.shared_state.phase_state.current_phase_index = 0
            else:
                self.shared_state.phase_state.current_phase_index += 1

            # Check if we ran out of phases
            if self.shared_state.phase_state.current_phase_index >= len(
                self.shared_state.phase_state.phase_queue
            ):
                self.shared_state.phase_state.current_phase_index = None
            self.shared_state.phase_state.is_running.clear()

    def opcontrol_legacy(self):
        self.max_rpm = 360

        # --- Tuning Variables ---
        SIGNIFICANT_CHANGE = (
            0.03  # Target threshold (how much a stick must move to update)
        )
        HEARTBEAT_INTERVAL = 15  # Max seconds to wait before refreshing active inputs

        # Track past values
        last_x = 0.0
        last_y = 0.0
        last_r = 0.0
        last_send_time = 0.0
        dx = 0.0
        dy = 0.0
        dr = 0.0

        while not self.queue_channels.kill_flag.is_set():
            with self.shared_state.drive_command_lock:
                x_movement = self.shared_state.drive_x
                y_movement = self.shared_state.drive_y
                r_movement = self.shared_state.drive_r

            # 1. Evaluate Delta Changes
            dx = abs(x_movement - last_x)
            dy = abs(y_movement - last_y)
            dr = abs(r_movement - last_r)

            # 2. Check if the robot is completely resting
            is_resting = x_movement == 0.0 and y_movement == 0.0 and r_movement == 0.0
            was_resting = last_x == 0.0 and last_y == 0.0 and last_r == 0.0

            # 3. Determine if an absolute network transmission is required
            current_time = time.time()
            time_since_last_send = current_time - last_send_time

            should_send = False

            if (
                dx > SIGNIFICANT_CHANGE
                or dy > SIGNIFICANT_CHANGE
                or dr > SIGNIFICANT_CHANGE
            ):
                # The user moved a stick significantly
                should_send = True
            elif is_resting != was_resting:
                # Transited from moving to absolute zero, or zero to moving
                should_send = True
            elif not is_resting and time_since_last_send >= HEARTBEAT_INTERVAL:
                # Sticks are held down constantly; refresh before the internal gRPC timeout cuts power
                should_send = True

            # 4. Process and execute command if approved
            if should_send:
                drive_tuple = self.engbot.calculate_mecanum_powers(
                    x_movement, y_movement, r_movement, self.max_rpm
                )

                self.engbot._sdk.mecanum_motor_control(
                    drive_tuple[0], drive_tuple[1], drive_tuple[2], drive_tuple[3]
                )

                # Cache historical state
                last_x = x_
