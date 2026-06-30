import logging
import time

import pygame  # for controller
from numpy.testing import print_assert_equal

# from ugot import ugot
import ns_robot
import ns_shared

logger = logging.getLogger(__name__)


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
        if not ns_shared.DEBUG_MODE:
            self.setup_engbot(engbot)
            # self.setup_sbbot(sbbot)

    def setup_engbot(self, engbot):
        logger.info("Attempting to connect ENGBot...")
        self.engbot.connect()
        # put shi here
        logger.info("Loading ENGBot models...")
        self.engbot._sdk.load_models(
            [
                "color_recognition",  # detects dominant colors
                "word_recognition",  # OCR: reads printed text
                "line_recognition",  # for line-following tasks
                "face_recognition",  # identifies registered faces by name
                "apriltag_qrcode",  # AprilTag recognition
            ]
        )
        # end

    def setup_sbbot(self, sbbot):

        logger.info("Attempting to connect SBBot...")
        self.sbbot.connect()

    def mainloop(self):
        """
        The main purpose of this slave mainloop is to listen to the process_manager mainloop,
        be a good boy, and just be easily killable by process_manager."""

        # very temporary just for testing please delete later

        while not self.queue_channels.kill_flag.is_set():
            self.sharedState.phase_state.is_running.wait()  # this will freeze the thread until it is running
            match self.sharedState.phase_state.phase_queue[
                self.sharedState.phase_state.current_phase_index
            ]:
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
                        f"The phase {
                            self.sharedState.phase_state.phase_queue[
                                self.sharedState.phase_state.current_phase_index
                            ]
                        } does not have a corresponding function tied to it in robot_controller!"
                    )
                    self.advance_phase()  # to make sure nothing softlocks
        # try and kill bots
        if not ns_shared.DEBUG_MODE:
            self.kill_bots()
        # now it shld die

    # def test(self):
    #     self.engbot._sdk.mecanum_motor_control(360, 360, 360, 360)
    #     # time.sleep(2)
    #     self.engbot._sdk.mecanum_move_speed_times(0, 80, 2, 0)
    #     self.engbot._sdk.mecanum_stop()
    #     time.sleep(10)

    def kill_bots(self):
        try:
            self.sbbot._sdk.balance_stop_balancing()
        except Exception as e:
            logger.exception(e)
            logger.warning("FAILED TO STOP SBBOT, WARNING!")
        try:
            self.engbot._sdk.mecanum_stop()
        except Exception as e:
            logger.exception(e)
            logger.warning("FAILED TO STOP SBBOT, WARNING!")

    def phase1(self):
        """sbb moves to april tag"""
        time.sleep(1)
        self.advance_phase()

    def phase2(self):
        got.face_recognition_add_name("Bad Guy")
        # Call red ball pickup code
        # stop at line
        # find villian at pos 1,2,3
        # align and throw at pos 1,2,3

        time.sleep(1)
        self.advance_phase()

    # with self.gui.shared_state.webcam_camera_frame_lock:
    #   frame = self.gui.shared_state.webcam_camera_frame
    # sharedState
    # shared_state
    # SharedState

    def phase2a(self):
        # call opcontrol portion
        self.opcontrol()
        self.advance_phase()

    def phase3(self):
        time.sleep(1)
        self.advance_phase()

    def phase4(self):
        time.sleep(1)
        self.advance_phase()

    def phase4a(self):
        # call opcontrol portion
        self.opcontrol()
        self.advance_phase()

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
                last_x = x_movement
                last_y = y_movement
                last_r = r_movement
                last_send_time = current_time

            # Keep your thread execution cycle polite to the CPU
            time.sleep(0.01)

    def opcontrol(self, joystick_control=True):
        """fall back option using mecanum_move_xyz"""
        if joystick_control:
            self.queue_channels.vibrate_flag.set()
        while not self.queue_channels.kill_flag.is_set():
            self.max_speed = 80  # cm/s dumb sdk lol
            self.max_rotation_speed = 280
            with self.sharedState.drive_command_lock:
                buttons = self.sharedState.controller_buttons
                x_movement = self.sharedState.drive_x
                y_movement = self.sharedState.drive_y
                r_movement = self.sharedState.drive_r

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
            r_movement = -int(
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
