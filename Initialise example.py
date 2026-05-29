"""
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

"""
