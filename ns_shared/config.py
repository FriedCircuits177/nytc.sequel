from ns_shared.terms import Phase

DEFAULT_TIMELINE_CONFIG = [
    Phase.Phase1,
    Phase.Phase2,
    Phase.Phase2A,
    Phase.Phase3,
    Phase.Phase4,
    Phase.Phase4A,
]

SBBOT_NAME = "UGOT_9499"
ENGBOT_NAME = "UGOT_1B20"
SBBOT_IP = "192.168.137.238"
ENGBOT_IP = "192.168.137.214"
TURBOJPEG_PATH = "C:/libjpeg-turbo-gcc64/bin/libturbojpeg.dll"
VILLAIN_JPEG_PATH = "villain.jpeg"
MEDIAPIPE_MODEL_PATH = "pose_landmarker_full.task"

CAMERA_FOCAL_LENGTH = 480  # tune ts please please please

DEBUG_MODE = False  # True if testing in no-bot mode
