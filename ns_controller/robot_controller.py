import logging
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
        self.sharedState = SharedState
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
        except Exception as e:
            logger.error(f"ENGBot initialization failed: {e}. Running offline.")

    def async_setup_sbbot(self):
        _ = ns_shared.construct_thread(self.setup_sbbot)
        _.start()

    def setup_sbbot(self):
        try:
            logger.info("Attempting to connect SBBot...")
            self.sbbot.connect()
        except Exception as e:
            logger.error(f"SBBot initialization failed: {e}. Running offline.")

    def mainloop(self):
        """Listens to the process_manager and executes ordered autonomous sequences."""
        while not self.queue_channels.kill_flag.is_set():
            self.sharedState.phase_state.is_running.wait()

            try:
                with self.sharedState.phase_state.lock:
                    current_idx = self.sharedState.phase_state.current_phase_index
                    if current_idx is None or current_idx >= len(
                        self.sharedState.phase_state.phase_queue
                    ):
                        self.sharedState.phase_state.is_running.clear()
                        continue
                    current_phase = self.sharedState.phase_state.phase_queue[
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

                with self.sharedState.phase_state.lock:
                    self.sharedState.phase_state.is_running.clear()
                    self.sharedState.phase_state.current_phase_index = None
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
                getattr(self.sharedState, "peripheral_sbbot_status", None)
                == ns_shared.PeripheralStatus.CONNECTED
            ):
                logger.exception(e)
                logger.warning("FAILED TO STOP SBBOT, WARNING!")
        try:
            self.engbot._sdk.mecanum_stop()
        except Exception as e:
            if (
                getattr(self.sharedState, "peripheral_engbot_status", None)
                == ns_shared.PeripheralStatus.CONNECTED
            ):
                logger.exception(e)
                logger.warning("FAILED TO STOP ENGBOT, WARNING!")

    # --- REVERTED ORIGINAL PHASE METHODS ---

    def phase1(self):
        """sbb moves to april tag"""
        self.sbbot.SBB_AP_centralization_approaching(
            distance=0.15, gap=20, fwd_spd=20, turn_spd=5
        )
        self.sbbot.SBB_charge_and_stop()
        self.advance_phase()
        # success so we return True
        return True

    def phase2(self):
        # Call red ball pickup code
        self.engbot.red_ball_pickup()
        # find villain
        self.engbot.search_villain()
        # align and throw at pos 1,2,3
        self.engbot.beat_up()

        logger.info("P2 done")

        time.sleep(1)
        self.advance_phase()

    def phase2a(self):
        # call opcontrol portion
        self.opcontrol()
        self.advance_phase()

    def phase3(self):
        self.opcontrol_pose()
        self.advance_phase()

    def phase4(self):
        time.sleep(1)
        self.advance_phase()

    def phase4a(self):
        # call opcontrol portion
        self.opcontrol()
        self.advance_phase()

    def you_can_crush_it_as_dry_as_a_bone(self):
        logging.info("you can kiss it you can break all the rules")

    def opcontrol_pose(self):
        self.queue_channels.pose_recog_active_flag.set()
        while not self.queue_channels.kill_flag.is_set():
            with self.sharedState.drive_command_lock:
                print(
                    f"{self.sharedState.drive_x},{self.sharedState.drive_y},{self.sharedState.drive_r}"
                )
                time.sleep(0.05)
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
            self.max_zoom_rpm = 360
            with self.sharedState.drive_command_lock:
                buttons = self.sharedState.controller_buttons
                x_movement = self.sharedState.drive_x
                y_movement = self.sharedState.drive_y
                r_movement = self.sharedState.drive_r
                r2 = self.sharedState.drive_r2

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
        with self.sharedState.phase_state.lock:
            if self.sharedState.phase_state.current_phase_index is None:
                self.sharedState.phase_state.current_phase_index = 0
            else:
                self.sharedState.phase_state.current_phase_index += 1

            # Check if we ran out of phases
            if self.sharedState.phase_state.current_phase_index >= len(
                self.sharedState.phase_state.phase_queue
            ):
                self.sharedState.phase_state.current_phase_index = None
            self.sharedState.phase_state.is_running.clear()

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
            with self.sharedState.drive_command_lock:
                x_movement = self.sharedState.drive_x
                y_movement = self.sharedState.drive_y
                r_movement = self.sharedState.drive_r

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
