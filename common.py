from enum import IntEnum, unique
from typing import List, Tuple

import cv2
import numpy as np
import math


@unique
class CocoPart(IntEnum):
    """Body part locations in the 'coordinates' list."""
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16


SKELETON_CONNECTIONS = [(0, 1, (210, 182, 247)), (0, 2, (127, 127, 127)), (1, 2, (194, 119, 227)),
                        (1, 3, (199, 199, 199)), (2, 4, (34, 189, 188)), (3, 5, (141, 219, 219)),
                        (4, 6, (207, 190, 23)), (5, 6, (150, 152, 255)), (5, 7, (189, 103, 148)),
                        (5, 11, (138, 223, 152)), (6, 8, (213, 176, 197)), (6, 12, (40, 39, 214)),
                        (7, 9, (75, 86, 140)), (8, 10, (148, 156, 196)), (11, 12, (44, 160, 44)),
                        (11, 13, (232, 199, 174)), (12, 14, (120, 187, 255)), (13, 15, (180, 119, 31)),
                        (14, 16, (14, 127, 255))]


def normalise(all_coordinates: List) -> List:
    """The normalization is a simple coordinate transformation done in two steps:

    1. Translation: All the key points are translated such that the nose key point becomes the origin of the coordinate
        system. This is achieved by subtracting the nose key points coordinates from all other key points.

    2. Scaling: The key points are scaled such that the distance between the left shoulder and right shoulder key point
        becomes 1. This is done by dividing all key points coordinates by the distance between the left and right
        shoulder key point.
    """
    norm_coords = []  # Hold the normalised coordinates for every frame

    # Iterate over every frame
    for coordinates in all_coordinates:
        # Step 1: Translate
        coordinates = [
            [coordinate[0] - coordinates[CocoPart.Nose.value][0], coordinate[1] - coordinates[CocoPart.Nose.value][1]]
            for coordinate in coordinates
        ]

        # Step 2: Scale
        dist = math.hypot(coordinates[CocoPart.LShoulder.value][0] - coordinates[CocoPart.RShoulder.value][0],
                          coordinates[CocoPart.LShoulder.value][1] - coordinates[CocoPart.RShoulder.value][1])
        coordinates = [[coordinate[0] / dist, coordinate[1] / dist] for coordinate in coordinates]

        norm_coords.append(coordinates)

    return norm_coords

def visualise(img: np.ndarray, keypoint_sets: List, width: int, height: int, tranX=0, tranY=0, vis_keypoints: bool = True,
              vis_skeleton: bool = False) -> np.ndarray:
    """Draw keypoints/skeleton on the output video frame."""
    if vis_keypoints or vis_skeleton:
        coords = keypoint_sets[0]
        if vis_skeleton:
            for p1i, p2i, color in SKELETON_CONNECTIONS:
                p1 = (int(coords[p1i][0] * width) + tranX, int(coords[p1i][1] * height) + tranY)
                p2 = (int(coords[p2i][0] * width) + tranX, int(coords[p2i][1] * height) + tranY)
                if p1 == (0, 0) or p2 == (0, 0):
                    continue
                cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)
        if vis_keypoints:
            for i, kps in enumerate(coords):
                # Scale up joint coordinate
                p = (int(kps[0] * width), int(kps[1] * height))
                # Joint wasn't detected
                if p == (0, 0):
                    continue
                cv2.circle(img=img, center=p, radius=5, color=(255, 255, 255), thickness=-1)

    return img


def write_on_image(img: np.ndarray, text: str, color: List) -> np.ndarray:
    """Write text at the top of the image."""
    # Add a white border to top of image for writing text
    img = cv2.copyMakeBorder(src=img,
                             top=int(0.25 * img.shape[0]),
                             bottom=0,
                             left=0,
                             right=0,
                             borderType=cv2.BORDER_CONSTANT,
                             dst=None,
                             value=[255, 255, 255])
    for i, line in enumerate(text.split('\n')):
        y = 30 + i * 30
        cv2.putText(img=img,
                    text=line,
                    org=(20, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color,
                    thickness=3)

    return img
