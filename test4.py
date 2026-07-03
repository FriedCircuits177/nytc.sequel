from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.137.205")
got.mechanical_joint_control(0, -77, 6, 1000)
got.mechanical_clamp_release()
print(got.mechanical_get_clamp_status())
got.mechanical_clamp_close()
print(got.mechanical_get_clamp_status())
