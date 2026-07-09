from ugot import ugot
import time

got = ugot.UGOT()
got.initialize("192.168.137.106")

while True:
    got.mecanum_stop()
