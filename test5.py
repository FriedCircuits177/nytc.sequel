import time

from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.137.106")

got.mechanical_joint_control(0, 90, 90, 1000)
time.sleep(1.5)
got.mechanical_clamp_close()
time.sleep(0.5)
#got.mechanical_joint_control(angle_to_turn,10,90,150)
got.mechanical_single_joint_control(2,30,150)
#time.sleep(0.01)
    #intentional

got.mechanical_single_joint_control(3,-45,100)
got.mechanical_clamp_release()
