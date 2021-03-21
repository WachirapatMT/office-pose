import csv
import argparse
import base64

from typing import List

import numpy as np
import openpifpaf
import torch
import cv2

from common import CocoPart, SKELETON_CONNECTIONS, write_on_image
from processor import Processor

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: Verify the args since they were changed in v0.10.0
    openpifpaf.decoder.cli(parser, force_complete_pose=True,
                           instance_threshold=0.2, seed_threshold=0.5)
    openpifpaf.network.nets.cli(parser)
    parser.add_argument('--resolution', default=1, type=float,
                        help=('Resolution prescale factor from 640x480. '
                              'Will be rounded to multiples of 16.'))
    parser.add_argument('--resize', default=None, type=str,
                        help=('Force input image resize. '
                              'Example WIDTHxHEIGHT.'))
    parser.add_argument('--video', default=None, type=str,
                        help='Path to the video file.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug messages and autoreload')
    parser.add_argument('--exercise', default='seated_right_knee_extension', type=str,
                        help='Exercise ID to perform.')

    vis_args = parser.add_argument_group('Visualisation')
    vis_args.add_argument('--joints', default=False, action='store_true',
                          help='Draw joint\'s keypoints on the output video.')
    vis_args.add_argument('--skeleton', default=False, action='store_true',
                          help='Draw skeleton on the output video.')
    vis_args.add_argument('--save-output', default=False, action='store_true',
                          help='Save the result in a video file.')
    vis_args.add_argument('--fps', default=20, type=int,
                          help='FPS for the output video.')
    vis_args.add_argument('--out-path', default='result.mp4', type=str,
                          help='Save the output video at the path specified. .avi file format.')
    vis_args.add_argument('--csv-path', default='keypoints.csv', type=str,
                          help='Save the pose coordinates into a CSV file at the path specified.')

    args = parser.parse_args()

    # Add args.device
    args.device = torch.device('cpu')

    return args

def write_to_csv(frame_number: int, humans: List, width: int, height: int, csv_fp: str):
    """Save keypoint coordinates of the *first* human pose identified to a CSV file.

    Coordinates are scaled to refer the resized image.

    Columns are in order frame_no, nose.(x|y|p), (l|r)eye.(x|y|p), (l|r)ear.(x|y|p), (l|r)shoulder.(x|y|p),
    (l|r)elbow.(x|y|p), (l|r)wrist.(x|y|p), (l|r)hip.(x|y|p), (l|r)knee.(x|y|p), (l|r)ankle.(x|y|p)

    l - Left side of the identified joint
    r - Right side of the identified joint
    x - X coordinate of the identified joint
    y - Y coordinate of the identified joint
    p - Probability of the identified joint

    :param frame_number: Frame number for the video file
    :param humans: List of human poses identified
    :param width: Width of the image
    :param height: Height of the image
    :param csv_fp: Path to the CSV file
    """
    # Use only the first human identified using pose estimation
    coordinates = humans[0]['coordinates'] if len(humans) > 0 else [None for _ in range(17)]

    # Final row that will be written to the CSV file
    row = [frame_number] + ['' for _ in range(51)]  # Number of coco points * 3 -> 17 * 3 -> 51

    # Update the items in the row for every joint
    # TODO: Value for joints not identified? Currently stored as 0
    for part in CocoPart:
        if coordinates[part] is not None:
            index = 1 + 3 * part.value  # Index at which the values for this joint would start in the final row
            row[index] = coordinates[part][0] * width
            row[index + 1] = coordinates[part][1] * height
            row[index + 2] = coordinates[part][2]

    with open(csv_fp, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)

def visualise(img: np.ndarray, keypoint_sets: List, width: int, height: int, vis_keypoints: bool = True,
              vis_skeleton: bool = False) -> np.ndarray:
    """Draw keypoints/skeleton on the output video frame."""
    if vis_keypoints or vis_skeleton:
        for keypoints in keypoint_sets:
            coords = keypoints['coordinates']

            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    p1 = (int(coords[p1i][0] * width), int(coords[p1i][1] * height))
                    p2 = (int(coords[p2i][0] * width), int(coords[p2i][1] * height))

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

def image_pose(image_path='one_leg_l.png'):
    args = cli()

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    resolution = 0.5
    width_height = (int(width * resolution // 16) * 16, int(height * resolution // 16) * 16)

    processor_singleton = Processor(width_height, args)
    keypoint_sets, scores, width_height = processor_singleton.single_image(
                b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8')
            )
    print('keypoint:::', keypoint_sets)

    keypoint_sets = np.array([[
  [4.7093248e-01, 2.0426853e-01, 8.6702675e-01],
  [4.8350826e-01, 1.9285889e-01, 9.0909678e-01],
  [4.5933738e-01, 1.9329509e-01, 7.9952747e-01],
  [5.0037909e-01, 2.1018410e-01, 6.8106771e-01],
  [4.4295478e-01, 2.0732497e-01, 8.4605628e-01],
  [5.1420957e-01, 2.8525087e-01, 9.5134568e-01],
  [4.2685834e-01, 2.8133684e-01, 1.2184765e+00],
  [5.2738458e-01, 1.5919007e-01, 8.1505233e-01],
  [4.1852579e-01, 1.7545088e-01, 7.9655707e-01],
  [4.8712268e-01, 8.4518030e-02, 7.3338878e-01],
  [4.5714667e-01, 8.1567958e-02, 1.0000000e-03],
  [5.0157762e-01, 5.1679271e-01, 5.8042961e-01],
  [4.4701675e-01, 4.9484652e-01, 7.6062948e-01],
  [6.2654567e-01, 6.2967056e-01, 4.0503103e-01],
  [4.7448894e-01, 7.3578298e-01, 3.6731312e-01],
  [4.9454674e-01, 7.3082601e-01, 1.0000000e-03],
  [4.9181211e-01, 8.8942599e-01, 4.6938220e-01]]])

    keypoint_sets = [{
        'coordinates': keypoints.tolist(),
        'detection_id': i,
        'score': score,
        'width_height': width_height,
    } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]

    # with open(args.csv_path, mode='w', newline='') as csv_file:
    #         csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         csv_writer.writerow(['frame_no',
    #                              'nose.x', 'nose.y', 'nose.prob',
    #                              'l.eye.x', 'l.eye.y', 'l.eye.prob',
    #                              'r.eye.x', 'r.eye.y', 'r.eye.prob',
    #                              'l.ear.x', 'l.ear.y', 'l.ear.prob',
    #                              'r.ear.x', 'r.ear.y', 'r.ear.prob',
    #                              'l.shoulder.x', 'l.shoulder.y', 'l.shoulder.prob',
    #                              'r.shoulder.x', 'r.shoulder.y', 'r.shoulder.prob',
    #                              'l.elbow.x', 'l.elbow.y', 'l.elbow.prob',
    #                              'r.elbow.x', 'r.elbow.y', 'r.elbow.prob',
    #                              'l.wrist.x', 'l.wrist.y', 'l.wrist.prob',
    #                              'r.wrist.x', 'r.wrist.y', 'r.wrist.prob',
    #                              'l.hip.x', 'l.hip.y', 'l.hip.prob',
    #                              'r.hip.x', 'r.hip.y', 'r.hip.prob',
    #                              'l.knee.x', 'l.knee.y', 'l.knee.prob',
    #                              'r.knee.x', 'r.knee.y', 'r.knee.prob',
    #                              'l.ankle.x', 'l.ankle.y', 'l.ankle.prob',
    #                              'r.ankle.x', 'r.ankle.y', 'r.ankle.prob',
    #                              ])

    # write_to_csv(frame_number=0, humans=keypoint_sets, width=width, height=height, csv_fp=args.csv_path)


    img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=True, vis_skeleton=True)

    cv2.imshow('Image window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_pose()
    # originalImage = cv2.imread('neck_bend_r.png')
    # flipImage = cv2.flip(originalImage, 1)
    # cv2.imshow('flip', flipImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('neck_bend_l.png', flipImage)