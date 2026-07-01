import logging
import threading

import ns_controller
import ns_perception
import ns_robot
import ns_shared

logger = logging.getLogger(__name__)

class ProcessManager:
    def __init__(
        self,
        SBBot: ns_robot.RobotHardware,
        ENGBot: ns_robot.RobotHardware,
        QueueChannels,
        SharedState,
    ):
        self.SBBot = SBBot
        self.ENGBot = ENGBot
        self.queue_channels = QueueChannels
        self.shared_state = SharedState
        self.robot_controller = ns_controller.RobotController(
            SBBot, ENGBot, QueueChannels, SharedState
        )
        self.sbbot_camera = ns_perception.Camera(
            self.SBBot,
            QueueChannels,
            SharedState,
            SharedState.sb_camera_frame,
            SharedState.sb_camera_frame_lock,
        )
        self.sbbot_camera_gui_processor = ns_perception.CameraGUIProcessor(
            QueueChannels,
            SharedState,
            SharedState.sb_camera_frame,
            SharedState.sb_camera_frame_lock,
            SharedState.sb_gui_camera_frame,
            SharedState.sb_gui_camera_frame_lock,
        )
        self.engbot_camera = ns_perception.Camera(
            self.ENGBot,
            QueueChannels,
            SharedState,
            SharedState.eng_camera_frame,
            SharedState.eng_camera_frame_lock,
        )
        self.engbot_camera_gui_processor = ns_perception.CameraGUIProcessor(
            QueueChannels,
            SharedState,
            SharedState.eng_camera_frame,
            SharedState.eng_camera_frame_lock,
            SharedState.eng_gui_camera_frame,
            SharedState.eng_gui_camera_frame_lock,
        )
        self.webcam = ns_perception.Webcam(
            QueueChannels,
            SharedState,
            SharedState.raw_webcam_camera_frame,
            SharedState.raw_webcam_camera_frame_lock,
        )
        self.webcam_processor = ns_perception.WebcamProcessor(
            QueueChannels,
            SharedState,
            SharedState.raw_webcam_camera_frame,
            SharedState.raw_webcam_camera_frame_lock,
            SharedState.webcam_camera_frame,
            SharedState.webcam_camera_frame_lock,
        )

        self.block_detection = ns_perception.BlockDetector(
            QueueChannels,
            SharedState,
        )
        self.ps4_driver = ns_controller.PS4ControllerDriver(QueueChannels, SharedState)
        #self.pose_recog = ns_perception.MediaPipePoseRecog(QueueChannels, SharedState)
        if ns_shared.DEBUG_MODE:
            self.threads = [
                ns_shared.construct_thread(self.robot_controller.mainloop),
                # ns_shared.construct_thread(self.sbbot_camera.mainloop),
                # ns_shared.construct_thread(self.sbbot_camera_gui_processor.mainloop),
                # ns_shared.construct_thread(self.engbot_camera.mainloop),
                # ns_shared.construct_thread(self.engbot_camera_gui_processor.mainloop),
                ns_shared.construct_thread(self.webcam.mainloop),
                ns_shared.construct_thread(self.webcam_processor.mainloop),
                ns_shared.construct_thread(self.block_detection.mainloop),
                ns_shared.construct_thread(self.ps4_driver.mainloop),
            ]
        else:
            self.threads = [
                ns_shared.construct_thread(self.robot_controller.mainloop),
                ns_shared.construct_thread(self.sbbot_camera.mainloop),
                ns_shared.construct_thread(self.sbbot_camera_gui_processor.mainloop),
                ns_shared.construct_thread(self.engbot_camera.mainloop),
                ns_shared.construct_thread(self.engbot_camera_gui_processor.mainloop),
                ns_shared.construct_thread(self.webcam.mainloop),
                ns_shared.construct_thread(self.webcam_processor.mainloop),
                ns_shared.construct_thread(self.block_detection.mainloop),
                ns_shared.construct_thread(self.ps4_driver.mainloop),
                #ns_shared.construct_thread(self.pose_recog.mainloop),
            ]

        for _ in self.threads:
            logger.info(
                f"Starting thread {_.name} ({self.threads.index(_) + 1}/{len(self.threads)})"
            )
            _.start()

    def mainloop(self):
        while not self.queue_channels.kill_flag.is_set():
            pass
        logger.info("Starting cleanup")
        # now do cleanup
        for _ in self.threads:
            _.join()
            logger.info(
                f"Thread {_.name} joined ({self.threads.index(_) + 1}/{len(self.threads)})"
            )
        # and now he dies peacefully with all his children
        logger.info("All threads joined!")
        logger.info("ProcessManager succesfully shutdown")
