import time

from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.137.106")
prefix = "villain"
counter = 4
while True:
    print(counter)
    got.face_recognition_add_name(f"{prefix}{counter}")
    counter += 1
