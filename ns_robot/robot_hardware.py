import logging

from ugot import ugot

logger = logging.getLogger(__name__)


class RobotHardware:
    """Wrapper for the ugot.UGOT instance to allow for custom methods."""

    def __init__(self, queue_channels, shared_state, name, type, ip):
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
        try:
            self._sdk.initialize(self.ip)
        except Exception as e:
            logger.exception(e)
            # oh no couldn't find it
            logger.error(f"COULD NOT CONNECT TO {self.type} ON THE NETWORK")
            return
        logger.info(f"Succesfully connected to {self.type} at {self.ip}")

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
