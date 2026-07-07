import logging
import threading
import time
from enum import Enum

import cv2
import numpy as np
from dearpygui import dearpygui as dpg

import ns_shared

logger = logging.getLogger(__name__)


class BaseWindow:
    def __init__(self, gui, title: str):
        self.gui = gui
        self.title = title
        self.window_tag = None

    def build(self):
        raise NotImplementedError

    def update(self):
        pass


class WebcamWindow(BaseWindow):
    def __init__(self, gui):
        super().__init__(gui, "Webcam")
        self.texture_tag = "webcam_texture"
        self.image_tag = "webcam_image"
        self.width = 640
        self.height = 480


    def build(self):
        with dpg.texture_registry():  # type: ignore
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=[0.0] * (self.width * self.height * 4),
                tag=self.texture_tag,
            )

        with dpg.window(
            label=self.title,
            width=self.width + 20,
            height=self.height + 40,
        ) as self.window_tag:  # type: ignore
            dpg.add_image(self.texture_tag, tag=self.image_tag)

    def update(self):
        # print("UPDATING")
        with self.gui.shared_state.webcam_camera_frame_lock:
            frame = self.gui.shared_state.webcam_camera_frame
        if frame is None:
            print("FRAME IS NONE")
            return

        dpg.set_value(self.texture_tag, frame.ravel())


class RobotCameraWindow(BaseWindow):
    def __init__(
        self, gui, title: str, frame_attr_name: str, frame_lock: threading.Lock
    ):
        """
        Args:
            gui: Main GUI instance reference.
            title: Title of the window (e.g., "SBBot Camera" or "ENGBot Camera").
            frame_attr_name: String name of the variable on shared_state (e.g., 'sb_gui_camera_frame').
            frame_lock: Reference to the corresponding threading.Lock object.
        """
        super().__init__(gui, title)
        # Unique tags based on title string to avoid DPG collisions
        self.texture_tag = f"{title.lower().replace(' ', '_')}_texture"
        self.image_tag = f"{title.lower().replace(' ', '_')}_image"
        self.width = 640
        self.height = 480

        self.frame_attr_name = frame_attr_name
        self.frame_lock = frame_lock

    def build(self):
        with dpg.texture_registry():  # type: ignore
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=[0.0] * (self.width * self.height * 4),
                tag=self.texture_tag,
            )

        with dpg.window(
            label=self.title,
            width=self.width + 20,
            height=self.height + 40,
        ) as self.window_tag:  # type: ignore
            dpg.add_image(self.texture_tag, tag=self.image_tag)

    def update(self):
        # Safely fetch the dynamic attribute name from shared_state via reflection
        with self.frame_lock:
            frame = getattr(self.gui.shared_state, self.frame_attr_name, None)

        if frame is None:
            return
        dpg.set_value(self.texture_tag, frame.ravel())


class PhaseTimeline(BaseWindow):
    def __init__(self, gui):
        super().__init__(gui, "Phase Timeline Controls")

        # UI Tags
        self.window_tag = "PhaseTimeline"
        self.track_drawlist = "timeline_track_drawlist"
        self.header_drawlist = "timeline_header_drawlist"
        self.context_menu_tag = "timeline_context_menu"

        # Layout Dimensions
        self.sidebar_width = 150
        self.total_width = 1000
        self.track_width = self.total_width - self.sidebar_width

        self.header_height = 16
        self.track_height = 60
        self.window_height = self.header_height + self.track_height + 60

        # Phase block dimensions
        self.phase_width = 110
        self.phase_height = 44
        self.phase_spacing = 6
        self.start_x_offset = 15

        # Interaction tracking state
        self.right_clicked_index = None

    def build(self):
        with dpg.window(
            label=self.title,
            tag=self.window_tag,
            width=self.total_width,
            height=self.window_height,
            pos=[0, 1000 - self.window_height],
            no_resize=True,
            no_collapse=True,
        ):  # type: ignore
            with dpg.group(horizontal=True, horizontal_spacing=0):  # type: ignore
                # --- LEFT SIDEBAR (CONTROLS) ---
                with dpg.child_window(
                    width=self.sidebar_width,
                    height=self.track_height + self.header_height + 15,
                    border=True,
                ):  # type: ignore
                    with dpg.group(horizontal=True):  # type: ignore
                        play_btn = dpg.add_button(label="PLAY", callback=self._cb_play)
                        with dpg.theme() as play_theme:  # type: ignore
                            with dpg.theme_component(dpg.mvButton):  # type: ignore
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_Button, (46, 139, 87, 255)
                                )
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_ButtonHovered, (60, 179, 113, 255)
                                )
                        dpg.bind_item_theme(play_btn, play_theme)

                        stop_btn = dpg.add_button(label="STOP", callback=self._cb_stop)
                        with dpg.theme() as stop_theme:  # type: ignore
                            with dpg.theme_component(dpg.mvButton):  # type: ignore
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_Button, (178, 34, 34, 255)
                                )
                                dpg.add_theme_color(
                                    dpg.mvThemeCol_ButtonHovered, (220, 20, 60, 255)
                                )
                        dpg.bind_item_theme(stop_btn, stop_theme)

                    dpg.add_spacer(height=4)

                    # Populating items from Enum definitions explicitly
                    phase_options = [p.value[0] for p in ns_shared.Phase]
                    dpg.add_combo(
                        items=phase_options,
                        default_value="+ Add Phase",
                        width=-1,
                        callback=self._cb_add_phase,
                    )

                # --- RIGHT WORKSPACE (VISUAL TIMELINE) ---
                with dpg.group(horizontal_spacing=0):  # type: ignore
                    dpg.add_drawlist(
                        width=self.track_width,
                        height=self.header_height,
                        tag=self.header_drawlist,
                    )
                    dpg.add_drawlist(
                        width=self.track_width,
                        height=self.track_height,
                        tag=self.track_drawlist,
                    )

        with dpg.popup(
            self.track_drawlist,
            mousebutton=dpg.mvMouseButton_Right,
            tag=self.context_menu_tag,
        ):  # type: ignore
            dpg.add_menu_item(label="Delete Phase", callback=self._cb_delete_phase)
            dpg.add_menu_item(label="Clear All Phases", callback=self._cb_clear_phase)

        with dpg.item_handler_registry() as registry:  # type: ignore
            dpg.add_item_clicked_handler(callback=self._handle_timeline_click)
        dpg.bind_item_handler_registry(self.track_drawlist, registry)

    def update(self):
        self.redraw()

    def redraw(self):
        dpg.delete_item(self.header_drawlist, children_only=True)
        dpg.delete_item(self.track_drawlist, children_only=True)

        state = self.gui.shared_state.phase_state

        with state.lock:
            queue = list(state.phase_queue)
            current_idx = (
                state.current_phase_index
                if state.current_phase_index is not None
                else 0
            )
            is_running = state.is_running.is_set()

        # Background track draw
        dpg.draw_rectangle(
            (0, 0),
            (self.track_width, self.track_height),
            fill=(30, 30, 30, 255) if not is_running else (22, 25, 32, 255),
            color=(50, 50, 50, 255),
            parent=self.track_drawlist,
        )

        x_offset = self.start_x_offset
        y_center = self.track_height // 2
        y1 = y_center - (self.phase_height // 2)
        y2 = y_center + (self.phase_height // 2)

        for idx, phase_enum in enumerate(queue):
            display_name = phase_enum.value[0]
            p_type = phase_enum.value[1]
            func_text = p_type.value.upper()

            is_autonomous = (
                p_type.name == "AUTONOMOUS"
                if hasattr(p_type, "name")
                else "Autonomous" in str(p_type)
            )
            is_pose = (
                p_type.name == "POSE"
                if hasattr(p_type, "name")
                else "Manual (Pose)" in str(p_type)
            )

            if is_running and idx < current_idx:
                fill_color = (65, 70, 75, 255)
                accent_color = (130, 140, 150, 180)
                text_color = (150, 150, 150, 180)

            elif is_running and idx == current_idx:
                if is_autonomous:
                    fill_color = (36, 111, 232, 255)
                    accent_color = (180, 220, 255, 255)
                elif is_pose:
                    fill_color = (138, 43, 226, 255)
                    accent_color = (230, 190, 255, 255)
                else:
                    fill_color = (220, 110, 10, 255)
                    accent_color = (255, 220, 170, 255)
                text_color = (255, 255, 255, 255)

            else:
                if is_autonomous:
                    fill_color = (75, 105, 150, 255)
                    accent_color = (190, 215, 240, 220)
                elif is_pose:
                    fill_color = (90, 60, 135, 255)
                    accent_color = (210, 180, 240, 220)
                else:
                    fill_color = (165, 115, 75, 255)
                    accent_color = (240, 210, 190, 220)
                text_color = (230, 235, 245, 255)

            dpg.draw_rectangle(
                (x_offset, y1),
                (x_offset + self.phase_width, y2),
                fill=fill_color,
                color=(15, 15, 15, 255)
                if idx != current_idx or not is_running
                else (255, 255, 255, 255),
                rounding=4.0,
                thickness=1.5 if (idx == current_idx and is_running) else 1.0,
                parent=self.track_drawlist,
            )

            dpg.draw_text(
                (x_offset + 6, y1 + 4),
                func_text,
                parent=self.track_drawlist,
                size=10,
                color=accent_color,
            )

            dpg.draw_text(
                (x_offset + 6, y1 + 16),
                display_name,
                parent=self.track_drawlist,
                size=14,
                color=text_color,
            )

            x_offset += self.phase_width + self.phase_spacing

        if len(queue) > 0:
            target_idx = current_idx
            target_idx = max(0, min(target_idx, len(queue)))
            block_left_x = self.start_x_offset + target_idx * (
                self.phase_width + self.phase_spacing
            )

            if is_running:
                playhead_x = (
                    (block_left_x + (self.phase_width // 2))
                    if target_idx < len(queue)
                    else (block_left_x - (self.phase_spacing // 2))
                )
            else:
                playhead_x = block_left_x - (self.phase_spacing // 2)

            dpg.draw_triangle(
                (playhead_x - 6, 2),
                (playhead_x + 6, 2),
                (playhead_x, self.header_height),
                fill=(235, 50, 50, 255),
                color=(235, 50, 50, 255),
                parent=self.header_drawlist,
            )
            dpg.draw_line(
                (playhead_x, 0),
                (playhead_x, self.track_height),
                color=(235, 50, 50, 220),
                thickness=1,
                parent=self.track_drawlist,
            )

    def _handle_timeline_click(self, sender, app_data):
        state = self.gui.shared_state.phase_state
        if state.is_running.is_set():
            return

        mouse_button = app_data[0]
        mouse_pos = dpg.get_drawing_mouse_pos()
        mx, my = mouse_pos[0], mouse_pos[1]

        with state.lock:
            queue_len = len(state.phase_queue)

        clicked_idx = None
        y_center = self.track_height // 2
        y1 = y_center - (self.phase_height // 2)
        y2 = y_center + (self.phase_height // 2)

        if y1 <= my <= y2:
            for i in range(queue_len):
                bx1 = self.start_x_offset + i * (self.phase_width + self.phase_spacing)
                bx2 = bx1 + self.phase_width
                if bx1 <= mx <= bx2:
                    clicked_idx = i
                    break

        if clicked_idx is not None:
            if mouse_button == dpg.mvMouseButton_Left:
                with state.lock:
                    state.current_phase_index = clicked_idx
                dpg.hide_item(self.context_menu_tag)
            elif mouse_button == dpg.mvMouseButton_Right:
                self.right_clicked_index = clicked_idx
                dpg.show_item(self.context_menu_tag)
        else:
            if mouse_button == dpg.mvMouseButton_Right:
                dpg.hide_item(self.context_menu_tag)

    def _cb_play(self, sender, app_data):
        state = self.gui.shared_state.phase_state
        with state.lock:
            if len(state.phase_queue) == 0:
                return
            if state.current_phase_index is None or state.current_phase_index >= len(
                state.phase_queue
            ):
                state.current_phase_index = 0
        state.is_running.set()
        logger.info("Automation Pipeline Signal: PLAY Set")

    def _cb_stop(self, sender, app_data):
        with self.gui.shared_state.phase_state.lock:
            self.gui.shared_state.phase_state.is_running.clear()
        self.gui.queue_channels.force_stop_phase_flag.set()
        logger.info(
            "Automation Pipeline Signal: Set gui.queue_channels.force_stop_phase_flag"
        )

    def _cb_add_phase(self, sender, app_data):
        display_name = dpg.get_value(sender)
        if display_name == "+ Add Phase":
            return

        phase_enum = self._get_phase_from_display_name(display_name)

        if phase_enum:
            state = self.gui.shared_state.phase_state
            with state.lock:
                if not state.is_running.is_set():
                    state.phase_queue.append(phase_enum)
                    logger.info(f"Queue updated: Appended {phase_enum}")

        dpg.set_value(sender, "+ Add Phase")

    def _cb_clear_phase(self, sender, app_data):
        state = self.gui.shared_state.phase_state
        with state.lock:
            if not state.is_running.is_set():
                state.phase_queue.clear()
                logger.info("Queue updated: Removed all phases")

        self.right_clicked_index = None

    def _cb_delete_phase(self, sender, app_data):
        if self.right_clicked_index is None:
            return
        state = self.gui.shared_state.phase_state
        with state.lock:
            if not state.is_running.is_set() and 0 <= self.right_clicked_index < len(
                state.phase_queue
            ):
                removed = state.phase_queue.pop(self.right_clicked_index)
                logger.info(
                    f"Queue updated: Removed index {self.right_clicked_index} ({removed})"
                )

                if state.current_phase_index is not None:
                    if state.current_phase_index > self.right_clicked_index:
                        state.current_phase_index -= 1
                    if state.current_phase_index >= len(state.phase_queue):
                        state.current_phase_index = (
                            len(state.phase_queue) - 1 if state.phase_queue else 0
                        )
        self.right_clicked_index = None

    def _get_phase_from_display_name(self, name: str) -> ns_shared.Phase | None:
        for phase in ns_shared.Phase:
            if phase.value[0] == name:
                return phase
        return None


# class PeripheralWindow(BaseWindow):
#     def __init__(self, gui):
#         super().__init__(gui, "Peripheral Connections")

#         # --- CONFIGURATION REGISTRY ---
#         # Explicitly maps each peripheral row to its own dedicated tracking variables and command queue.
#         self.peripherals_config = [
#             {
#                 "name": f"Self-Balancing Bot ({ns_shared.SBBOT_NAME})",
#                 "id": "sb_bot",
#                 "state_attr": "peripheral_sbbot_status",  # Variable in SharedState
#                 "lock_attr": "peripheral_sbbot_status_lock",  # Lock in SharedState
#                 "queue_attr": "peripheral_sbbot_command_queue",  # Dedicated Queue attribute name in QueueChannels
#             },
#             {
#                 "name": f"Engineer Bot ({ns_shared.ENGBOT_NAME})",
#                 "id": "eng_bot",
#                 "state_attr": "peripheral_engbot_status",  # Variable in SharedState
#                 "lock_attr": "peripheral_engbot_status_lock",  # Lock in SharedState
#                 "queue_attr": "peripheral_engbot_command_queue",  # Dedicated Queue attribute name in QueueChannels
#             },
#             {
#                 "name": "Controller",
#                 "id": "controller",
#                 "state_attr": "peripheral_controller_status",  # Variable in SharedState
#                 "lock_attr": "peripheral_controller_status_lock",  # Lock in SharedState
#                 "queue_attr": "peripheral_controller_command_queue",  # Dedicated Queue attribute name in QueueChannels
#             },
#             # Pseudocode Example for future expansion:
#             # {
#             #     "name": "ENGBot Manipulator Arm",
#             #     "id": "eng_arm",
#             #     "state_attr": "eng_arm_alive",
#             #     "lock_attr": "eng_arm_lock",
#             #     "queue_attr": "eng_arm_command_queue"
#             # }
#         ]

#     def build(self):
#         with dpg.window(
#             label=self.title, width=550, height=300, no_collapse=True
#         ) as self.window_tag:  # type: ignore
#             # Using a table layout for pixel-perfect vertical column alignment
#             with dpg.table(header_row=True, borders_innerH=True, borders_innerV=True):  # type: ignore
#                 dpg.add_table_column(label="Device Name", width_stretch=True)
#                 dpg.add_table_column(
#                     label="Status", width_fixed=True, init_width_or_weight=130
#                 )
#                 dpg.add_table_column(
#                     label="Action", width_fixed=True, init_width_or_weight=110
#                 )

#                 for item in self.peripherals_config:
#                     device_id = item["id"]

#                     with dpg.table_row():  # type: ignore
#                         # Column 1: Device Descriptive Name
#                         dpg.add_text(item["name"])

#                         # Column 2: Status Text Label (Colored programmatically in update)
#                         dpg.add_text("Unknown", tag=f"p_status_{device_id}")

#                         # Column 3: Contextual Action Toggle Button
#                         dpg.add_button(
#                             label="Connect",
#                             tag=f"p_btn_{device_id}",
#                             callback=self._cb_toggle_connection,
#                             user_data=item,  # Pass the specific row config directly to callback
#                         )

#     def update(self):
#         """Polls SharedState variables safely and refreshes UI values dynamically."""
#         for item in self.peripherals_config:
#             device_id = item["id"]

#             # 1. Dynamically retrieve the lock and variable reference from SharedState
#             lock = getattr(self.gui.shared_state, item["lock_attr"], None)
#             if lock is None:
#                 continue

#             with lock:
#                 raw_status = getattr(self.gui.shared_state, item["state_attr"], False)

#             # 2. Normalize status if the backend uses a simple boolean value
#             if isinstance(raw_status, bool):
#                 if raw_status:
#                     status_enum = ns_shared.PeripheralStatus.CONNECTED
#                 else:
#                     status_enum = ns_shared.PeripheralStatus.DISCONNECTED
#             else:
#                 status_enum = raw_status

#             # 3. Map status state properties to textual UI elements and color weights
#             if status_enum == ns_shared.PeripheralStatus.CONNECTED:
#                 text_val = "Connected"
#                 text_color = (50, 220, 100, 255)  # Vibrant Green
#                 btn_label = "Disconnect"
#             elif status_enum == ns_shared.PeripheralStatus.CONNECTING:
#                 text_val = "Connecting..."
#                 text_color = (255, 165, 0, 255)  # Orange
#                 btn_label = "Disconnect"
#             else:
#                 text_val = "Disconnected"
#                 text_color = (220, 50, 50, 255)  # Red
#                 btn_label = "Connect"

#             # 4. Push safe UI configuration mutations back directly to DearPyGUI layout elements
#             status_tag = f"p_status_{device_id}"
#             btn_tag = f"p_btn_{device_id}"

#             if dpg.does_item_exist(status_tag):
#                 dpg.set_value(status_tag, text_val)
#                 dpg.configure_item(status_tag, color=text_color)

#             if dpg.does_item_exist(btn_tag):
#                 dpg.configure_item(btn_tag, label=btn_label)

#     def _cb_toggle_connection(self, sender, app_data, user_data):
#         """Universal dispatcher tracking current context state to flag backend actions."""
#         item = user_data
#         device_id = item["id"]

#         # 1. Pull live state value to evaluate target action logic safely
#         lock = getattr(self.gui.shared_state, item["lock_attr"])
#         with lock:
#             raw_status = getattr(self.gui.shared_state, item["state_attr"], False)

#         # 2. Evaluate current state to determine inverted command
#         if (
#             raw_status is True
#             or raw_status == ns_shared.PeripheralStatus.CONNECTED
#             or raw_status == ns_shared.PeripheralStatus.CONNECTING
#         ):
#             target_command = ns_shared.PeripheralConnectionCommand.DISCONNECT
#         else:
#             target_command = ns_shared.PeripheralConnectionCommand.CONNECT

#         # 3. Dynamically resolve the dedicated queue for this individual backend device
#         target_queue = getattr(self.gui.queue_channels, item["queue_attr"], None)

#         if target_queue is not None:
#             logger.info(
#                 f"GUI Command Sent: Requesting {target_command.name} via queue '{item['queue_attr']}'"
#             )
#             # Put the explicit command directly into the target device's private channel
#             target_queue.put({"command": target_command})
#         else:
#             logger.error(
#                 f"Failed to dispatch command: Queue channel '{item['queue_attr']}' not found in QueueChannels!"
#             )


class RobotCameraCycleWindow(BaseWindow):
    def __init__(self, gui, camera_configs: list[dict]):
        initial_title = (
            camera_configs[0]["caption"] if camera_configs else "Camera Cycle"
        )
        super().__init__(gui, initial_title)

        self.camera_configs = camera_configs
        self.current_index = 0

        instance_id = id(self)
        self.window_tag = f"cycle_window_{instance_id}"
        self.texture_tag = f"cycle_texture_{instance_id}"
        self.image_tag = f"cycle_image_{instance_id}"
        self.handler_registry_tag = f"cycle_handlers_{instance_id}"

        self.width = 640
        self.height = 480

        # ADD THIS LINE HERE:
        self.last_triangle_state = False

    def build(self):
        if not self.camera_configs:
            logger.error(
                "RobotCameraCycleWindow: Initialized with an empty config list!"
            )
            return

        # Setup dynamic texture registry matching standard sizing
        with dpg.texture_registry():  # type: ignore
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=[0.0] * (self.width * self.height * 4),
                tag=self.texture_tag,
            )

        # Setup window using the dynamic label of the current camera index
        with dpg.window(
            label=self.camera_configs[self.current_index]["caption"],
            tag=self.window_tag,
            width=self.width + 20,
            height=self.height + 40,
        ):  # type: ignore
            dpg.add_image(self.texture_tag, tag=self.image_tag)

        # Create and bind an item handler registry directly to the image feed item
        with dpg.item_handler_registry(tag=self.handler_registry_tag):  # type: ignore
            dpg.add_item_clicked_handler(callback=self._handle_feed_click)
        dpg.bind_item_handler_registry(self.image_tag, self.handler_registry_tag)

        # Sync the initial threading state for all cameras on application boot
        self._sync_camera_thread_states()

    def _handle_feed_click(self, sender, app_data):
        # app_data structured as: [mouse_button, status]
        mouse_button = app_data[0]

        if mouse_button == dpg.mvMouseButton_Left:
            # Safely cycle index through your entire configurations array length
            self.current_index = (self.current_index + 1) % len(self.camera_configs)

            # Mutate window caption property live without changing the absolute target tags
            new_config = self.camera_configs[self.current_index]
            dpg.configure_item(self.window_tag, label=new_config["caption"])

            # Dynamically flip the threading execution flags on the SharedState object
            self._sync_camera_thread_states()

            logger.info(f"Camera window cycled to view: {new_config['caption']}")

    def _sync_camera_thread_states(self):
        """Iterates through camera configurations and flips the threading.Event states."""
        if not hasattr(self.gui, "queue_channels") or self.gui.queue_channels is None:
            logger.warning(
                "RobotCameraCycleWindow: queue_channels not initialized yet. Skipping sync."
            )
            return

        for idx, config in enumerate(self.camera_configs):
            attr_name = config.get("active_attr_name")
            if attr_name:
                # Retrieve the event object from queue_channels safely
                event = getattr(self.gui.queue_channels, attr_name, None)

                if event is None:
                    # This log will catch if your front-end is building too fast for your backend
                    logger.error(
                        f"RobotCameraCycleWindow: Threading Event '{attr_name}' NOT found in queue_channels!"
                    )
                    continue

                if idx == self.current_index:
                    event.set()  # Automatically unblocks index 0 on startup
                    logger.info(
                        f"Set threading flag active for: {config['caption']} ({attr_name})"
                    )
                else:
                    event.clear()  # Ensure all other cameras are paused
                    logger.info(
                        f"Cleared threading flag for: {config['caption']} ({attr_name})"
                    )

    def update(self):
        if not self.camera_configs:
            return

        # --- CONTROLLER TRIANGLE EDGE DETECTOR ---
        with self.gui.shared_state.drive_command_lock:
            current_triangle = self.gui.shared_state.controller_buttons.get(
                "triangle", False
            )

        # Detect button release: was pressed last frame, but released this frame
        if self.last_triangle_state and not current_triangle:
            self.current_index = (self.current_index + 1) % len(self.camera_configs)
            new_config = self.camera_configs[self.current_index]
            dpg.configure_item(self.window_tag, label=new_config["caption"])
            self._sync_camera_thread_states()
            logger.info(
                f"Controller Event: TRIANGLE -> Cycled camera window to: {new_config['caption']}"
            )

        # Save the current state to compare against on the next frame
        self.last_triangle_state = current_triangle
        # ----------------------------------------

        current_config = self.camera_configs[self.current_index]

        # --- STARTUP FALLBACK CHECK ---
        # If the backend event was missed during build(), force activate it here
        attr_name = current_config.get("active_attr_name")
        if attr_name and hasattr(self.gui, "queue_channels"):
            event = getattr(self.gui.queue_channels, attr_name, None)
            if event and not event.is_set():
                logger.info(
                    f"Fallback activation triggered for {current_config['caption']}"
                )
                self._sync_camera_thread_states()
        # ------------------------------

        lock = current_config["frame_lock"]
        frame_attr = current_config["frame_attr_name"]

        # Resource-safe lock acquisition mirroring previous thread structures
        with lock:
            frame = getattr(self.gui.shared_state, frame_attr, None)

        if frame is None:
            return

        dpg.set_value(self.texture_tag, frame.ravel())


class SaveLayoutWindow(BaseWindow):
    def __init__(self, gui):
        super().__init__(gui, "Layout Manager")
        self.window_tag = "temporary_layout_window"

    def build(self):
        # Prevent the window from being saved to the .ini file itself,
        # and configure it as a small, clean utility window.
        with dpg.window(
            label=self.title,
            tag=self.window_tag,
            width=200,
            height=80,
            no_resize=True,
            no_saved_settings=True,  # <--- Crucial line!
        ):  # type: ignore
            dpg.add_button(
                label="Save Current Layout",
                callback=self._cb_save_layout,
                width=-1,  # Fill the window width horizontally
                height=-1,  # Fill the window height vertically
            )

    def _cb_save_layout(self, sender, app_data):
        try:
            # Serializes the active layout states into dpg_default_layout.ini
            dpg.save_init_file("dpg_default_layout.ini")
            logger.info(
                "GUI layout successfully serialized to 'dpg_default_layout.ini'"
            )
        except Exception as e:
            logger.error(f"Failed to save layout settings: {e}")


class GUI:
    def __init__(self, QueueChannels, SharedState):
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.windows = []

    def build(self):
        dpg.create_context()
        dpg.configure_app(docking=True, docking_space=True)

        with dpg.font_registry():  # type: ignore
            try:
                default_font = dpg.add_font("ns_gui/assets/Inter_18pt-Regular.ttf", 18)
                dpg.bind_font(default_font)
            except Exception:
                pass
        camera_rotation_setup = [
            {
                "caption": "SBBot Camera",
                "frame_attr_name": "sb_gui_camera_frame",
                "frame_lock": self.shared_state.sb_gui_camera_frame_lock,
                "active_attr_name": "sbbot_camera_active_flag",
            },
            {
                "caption": "ENGBot Camera",
                "frame_attr_name": "eng_gui_camera_frame",
                "frame_lock": self.shared_state.eng_gui_camera_frame_lock,
                "active_attr_name": "engbot_camera_active_flag",
            },
            # You can append unlimited dictionaries here dynamically later!
        ]
        # Initializing the distinct windows using our dynamic references
        self.windows.append(WebcamWindow(self))

        # self.windows.append(
        #     RobotCameraWindow(
        #         gui=self,
        #         title="SBBot Camera",
        #         frame_attr_name="sb_gui_camera_frame",
        #         frame_lock=self.shared_state.sb_gui_camera_frame_lock,
        #     )
        # )

        # self.windows.append(
        #     RobotCameraWindow(
        #         gui=self,
        #         title="ENGBot Camera",
        #         frame_attr_name="eng_gui_camera_frame",
        #         frame_lock=self.shared_state.eng_gui_camera_frame_lock,
        #     )
        # )

        self.windows.append(PhaseTimeline(self))
        self.windows.append(
            RobotCameraCycleWindow(self, camera_configs=camera_rotation_setup)
        )
        # self.windows.append(SaveLayoutWindow(self))
        for window in self.windows:
            window.build()

        dpg.create_viewport(
            title="nytc.sequel",
            width=1920,
            height=1080,
            resizable=False,
            x_pos=0,
            y_pos=0,
        )
        dpg.configure_app(init_file="dpg_default_layout.ini")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

    def mainloop(self):
        last = time.time()
        frames = 0

        while dpg.is_dearpygui_running():
            for window in self.windows:
                window.update()

            dpg.render_dearpygui_frame()
            frames += 1

            if time.time() - last > 1:
                # print("GUI FPS:", frames)
                frames = 0
                last = time.time()

        dpg.destroy_context()
