from ugot import ugot
import time
import importlib
import cv2
import numpy as numpy
import pose_yolo
from pose_yolo import run_pose_control_inline
import pygame
import sys

got = ugot.UGOT()

got.load_models(
    [
        "color_recognition",  # detects dominant colors
        "word_recognition",  # OCR: reads printed text
        "line_recognition",  # for line-following tasks
        "face_recognition",  # identifies registered faces by name
        "apriltag_qrcode",  # AprilTag recognition
    ]
)

got.set_track_recognition_line(0)
got.open_camera()

print("Initialise done!")


def get_face_direction():
    print("finding face...")
    direction = "ERROR"

    return direction


def throw():
    angle_offset = 25
    J1_angle = 0
    direction = get_face_direction()
    print(f"Direction is:", direction, "!")
    if direction == "LEFT":
        J1_angle = -angle_offset
    elif direction == "RIGHT":
        J1_angle = angle_offset
    elif direction == "CENTER":
        J1_angle = 0
    elif direction == "ERROR":
        print("ERROR, no face found! Warning!")
        sys.exit()
    else:
        print("this shouldn't happen")
        print("direction has no value")
        print("why.")
        sys.exit()

    print("Preparing!")
    got.mechanical_joint_control(J1_angle, 110, -45, 2000)
    time.sleep(1)
    print("Throwing!")
    got.mechanical_joint_control(J1_angle, 20, 0, 750)
    got.mechanical_clamp_release()
    got.mechanical_joint_control(J1_angle, 0, 0, 100)
    print("Did I hit?")
