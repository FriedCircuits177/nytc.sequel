import time

from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.137.214")

got.mechanical_clamp_release()
time.sleep(1)
got.mechanical_clamp_close()
