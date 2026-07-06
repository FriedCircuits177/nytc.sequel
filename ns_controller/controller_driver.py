import logging
import os
import queue
import time

import pygame

import ns_shared
from ns_shared import QueueChannels, SharedState, shared_state

logger = logging.getLogger(__name__)

# Force HIDAPI driver for robust rumble support on dualshock
os.environ["SDL_JOYSTICK_HIDAPI_PS4_RUMBLE"] = "1"


class PS4ControllerDriver:
    def __init__(self, queue_channels: QueueChannels, shared_state: SharedState):
        self.queue_channels = queue_channels
        self.shared_state = shared_state
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.display.set_mode((1, 1))
        pygame.joystick.init()

        self.joystick = None
        self.deadzone = 0.12  # Filters out analog drift when sticks are resting

        with self.shared_state.drive_command_lock:
            # Seed our dynamic properties explicitly
            self.controller_buttons = self.shared_state.controller_buttons.copy()

        # FIXED: Ensure this is a discrete instance snapshot, not a shared pointer reference
        self.controller_buttons_last = self.controller_buttons.copy()

    def init_controller(self) -> bool:
        """Attempts to discover and lock onto a connected PS4 controller."""
        # with self.shared_state.peripheral_controller_status_lock:
        #     self.shared_state.peripheral_controller_status = (
        #         ns_shared.PeripheralStatus.CONNECTING
        #     )
        pygame.joystick.quit()

        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            logger.warning("No joysticks detected! Retrying...")
            return False

        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(
                f"Successfully locked onto controller: {self.joystick.get_name()}"
            )
            self.joystick.rumble(0.5, 0.5, 250)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize joystick device: {e}")
            self.joystick = None
            with self.shared_state.peripheral_controller_status_lock:
                self.shared_state.peripheral_controller_status = (
                    ns_shared.PeripheralStatus.DISCONNECTED
                )
            return False

    def filter_deadzone(self, value: float) -> float:
        """Clips small analog stick outputs to prevent the robot from creeping."""
        if abs(value) < self.deadzone:
            return 0.0
        # Smooth out scale after deadzone reduction
        return (value - (self.deadzone * (1 if value > 0 else -1))) / (
            1.0 - self.deadzone
        )

    def check_if_released(self, button_name: str) -> bool:
        # Avoid KeyError if a new button hasn't been added into one of the dicts yet
        current = self.controller_buttons.get(button_name, False)
        last = self.controller_buttons_last.get(button_name, False)
        return (not current) and last

    def joystick_flag_send(self):
        """Checks edge transitions to switch flags and modify phase state on button release."""
        state = self.shared_state.phase_state

        # --- CIRCLE BUTTON: START/PLAY PHASE ---
        if self.check_if_released("circle"):
            with state.lock:
                if len(state.phase_queue) > 0:
                    if (
                        state.current_phase_index is None
                        or state.current_phase_index >= len(state.phase_queue)
                    ):
                        state.current_phase_index = 0
                    state.is_running.set()
                    logger.info("Controller Event: CIRCLE -> Pipeline STARTED")

        # --- SQUARE BUTTON: STOP PHASE ---
        elif self.check_if_released("square"):
            with state.lock:
                state.is_running.clear()
            self.queue_channels.force_stop_phase_flag.set()
            logger.info("Controller Event: SQUARE -> Pipeline STOPPED")

        # --- D-LEFT ARROW: DECREMENT TARGET PHASE INDEX ---
        elif self.check_if_released("d_left"):
            with state.lock:
                # Only allow navigation if the automation pipeline isn't actively running
                if not state.is_running.is_set() and state.phase_queue:
                    if state.current_phase_index is None:
                        state.current_phase_index = 0
                    else:
                        state.current_phase_index = max(
                            0, state.current_phase_index - 1
                        )
                    logger.info(
                        f"Controller Event: D-Left -> Target Index: {state.current_phase_index}"
                    )

        # --- D-RIGHT ARROW: INCREMENT TARGET PHASE INDEX ---
        elif self.check_if_released("d_right"):
            with state.lock:
                # Only allow navigation if the automation pipeline isn't actively running
                if not state.is_running.is_set() and state.phase_queue:
                    if state.current_phase_index is None:
                        state.current_phase_index = 0
                    else:
                        state.current_phase_index = min(
                            len(state.phase_queue) - 1, state.current_phase_index + 1
                        )
                    logger.info(
                        f"Controller Event: D-Right -> Target Index: {state.current_phase_index}"
                    )

        # FIXED: Create a brand new copy snapshot here every time
        self.controller_buttons_last = self.controller_buttons.copy()

    def mainloop(self):
        """Dedicated loop pumping hardware events and publishing vectors to state."""
        while not self.queue_channels.kill_flag.is_set():
            # Connection loop if the hardware drops out\
            # print("CONTROLLER DRIVER ALIVE")
            # if self.queue_channels.peripheral_controller_command_queue.full():
            #     if (
            #         self.queue_channels.peripheral_controller_command_queue.get()
            #         == ns_shared.PeripheralConnectionCommand.DISCONNECT
            #     ):
            #         # gui asked for dc so we forcefully dc
            #         print("disconnecting")
            #         self.joystick = None
            #         with self.shared_state.peripheral_controller_status_lock:
            #             self.shared_state.peripheral_controller_status = (
            #                 ns_shared.PeripheralStatus.DISCONNECTED
            #             )
            if not self.joystick:
                if not self.init_controller():
                    time.sleep(2.0)
                    continue
                # initialisation succesful
                # with self.shared_state.peripheral_controller_status_lock:
                #     self.shared_state.peripheral_controller_status = (
                #         ns_shared.PeripheralStatus.CONNECTED
                #     )

            try:
                try:
                    # Check if any backend function or thread requested a custom rumble
                    small_mag, large_mag, duration = (
                        self.queue_channels.vibrate_flag.get_nowait()
                    )
                    # Pygame expects float values between 0.0 and 1.0
                    self.joystick.rumble(small_mag, large_mag, duration)
                except queue.Empty:
                    pass  # No custom rumble requests in queue

                pygame.event.pump()  # Flushes the OS event message registers

                # 1. Capture Analog Stick Movements
                raw_l2 = self.joystick.get_axis(4)
                raw_x = self.joystick.get_axis(0)
                raw_y = -self.joystick.get_axis(1)
                raw_r = -self.joystick.get_axis(2)

                if raw_l2 > 0.4:
                    # Scale intensity linearly with trigger pressure
                    intensity = (
                        (raw_l2 + 1.0) / 2.0
                    ) * 0.4  # Max 25% strength for subtlety
                    # Small motor = high frequency (buzzing), Large motor = low frequency (thumping)
                    self.joystick.rumble(intensity, 0.0, 50)

                # print(f"{raw_x},{raw_y},{raw_r}")
                # 2. Process math limits
                x_vel = self.filter_deadzone(raw_x)
                y_vel = self.filter_deadzone(raw_y)
                r_vel = self.filter_deadzone(raw_r)

                # Capture digital transitions safely
                self.controller_buttons["cross"] = bool(self.joystick.get_button(0))
                self.controller_buttons["circle"] = bool(self.joystick.get_button(1))
                self.controller_buttons["square"] = bool(self.joystick.get_button(2))
                self.controller_buttons["triangle"] = bool(self.joystick.get_button(3))

                self.controller_buttons["d_up"] = bool(self.joystick.get_button(11))
                self.controller_buttons["d_down"] = bool(self.joystick.get_button(12))
                self.controller_buttons["d_left"] = bool(self.joystick.get_button(13))
                self.controller_buttons["d_right"] = bool(self.joystick.get_button(14))

                # Dispatches releases checking old states vs new states
                self.joystick_flag_send()

                # 3. Safely update coordinates in shared state
                with self.shared_state.drive_command_lock:
                    self.shared_state.drive_x = x_vel
                    self.shared_state.drive_y = y_vel
                    self.shared_state.drive_r = r_vel
                    self.shared_state.drive_l2 = raw_l2
                    self.shared_state.controller_buttons = (
                        self.controller_buttons.copy()
                    )

            except pygame.error as e:
                logger.error(f"Controller communication error (disconnected?): {e}")
                self.joystick = (
                    None  # Resets state to force re-connection sequence next loop
                )
                # with self.shared_state.peripheral_controller_status_lock:
                #     self.shared_state.peripheral_controller_status = (
                #         ns_shared.PeripheralStatus.DISCONNECTED
                #     )

            # Keep CPU happy: ~100Hz loop rate cuts thread cost down close to 0%
            time.sleep(0.01)  # 10ms I think
