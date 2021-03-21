import argparse
import base64
import csv
import time

from typing import List
from itertools import chain

import cv2
import numpy as np
import openpifpaf
import torch
from scipy.spatial.distance import euclidean

from common import CocoPart, SKELETON_CONNECTIONS, write_on_image, visualise, normalise
from processor import Processor
from exercise import SIDE_BEND_LEFT, SIDE_BEND_RIGHT, NECK_BEND_L, NECK_BEND_R, ONE_LEG_L, ONE_LEG_R

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    openpifpaf.decoder.cli(parser, force_complete_pose=True,
                           instance_threshold=0.2, seed_threshold=0.5)
    openpifpaf.network.nets.cli(parser)
    parser.add_argument('--resolution', default=0.4, type=float,
                        help=('Resolution prescale factor from 640x480. '
                              'Will be rounded to multiples of 16.'))
    parser.add_argument('--video', default=None, type=str,
                        help='Path to the video file.')

    vis_args = parser.add_argument_group('Visualisation')
    vis_args.add_argument('--joints', default=False, action='store_true',
                          help='Draw joint\'s keypoints on the output video.')
    vis_args.add_argument('--skeleton', default=False, action='store_true',
                          help='Draw skeleton on the output video.')
    vis_args.add_argument('--compare', default=False, action='store_true',
                          help='Compare between user pose and the chosen exercise')

    # vis_args.add_argument('--save-output', default=False, action='store_true',
    #                       help='Save the result in a video file.')
    # vis_args.add_argument('--fps', default=25, type=int,
    #                       help='FPS for the output video.')
    # vis_args.add_argument('--out-path', default='pose_result.mp4', type=str,
    #                       help='Save the output video at the path specified. .avi file format.')

    args = parser.parse_args()

    # Add args.device
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    return args

def main():
    args = cli()
    
    # exercise = SIDE_BEND_LEFT
    # exercise = SIDE_BEND_RIGHT
    # exercise = NECK_BEND_R
    # exercise = NECK_BEND_L
    # exercise = ONE_LEG_R
    exercise = ONE_LEG_L

    # Video source
    if args.video is None:
        print('Video source: default webcam')
        cam = cv2.VideoCapture(0)
    elif args.video == '1':
        print('Video source: mobile camera')
        cam = cv2.VideoCapture(1)
    else:
        print(f'Video source: {args.video}')
        cam = cv2.VideoCapture(args.video)

    ret_val, img = cam.read()
    height, width = img.shape[:2]
    width_height = (int(width * args.resolution // 16) * 16, int(height * args.resolution // 16) * 16)
    print(f'Resize image from {(width, height)} to {width_height}')
    processor_singleton = Processor(width_height, args)

    # For FPS calculation
    start = time.time()
    frame = 0

    while True:
        frame += 1

        ret_val, img = cam.read()
        if img is None: break
        if not ret_val:
            task_finished = True
            continue

        # Press esc to exit
        if cv2.waitKey(1) == 27:
            end = time.time()
            print(f'FPS: {frame/(end-start)}')
            break

        keypoint_sets, scores, width_height = processor_singleton.single_image(
            b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8')
        )

        if not args.compare:
            img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=args.joints,
                            vis_skeleton=args.skeleton)

        try:
            pose0 = [list(map(lambda x: [x[0], x[1]], keypoint_sets[0]))]
            pose1 = [list(map(lambda x: [x[0], x[1]], exercise[0]))]

            pose0 = normalise(pose0)
            pose1 = normalise(pose1)

            ### compare ###
            if args.compare:
                # comment out to use real image as a background
                img = np.ones((480,640,3), np.uint8) * 255
                pose00 = [list(map(lambda x: [x[0], x[1], 1], pose0[0]))]
                pose11 = [list(map(lambda x: [x[0], x[1], 1], pose1[0]))]

                img = visualise(img=img, keypoint_sets=pose00, width=width//12, height=height//12, tranX=400, tranY=110, vis_keypoints=args.joints,
                            vis_skeleton=args.skeleton)
                img = visualise(img=img, keypoint_sets=pose11, width=width//12, height=height//12, tranX=150, tranY=110, vis_keypoints=args.joints,
                            vis_skeleton=args.skeleton)
            ###############

            score = euclidean(list(chain(*pose1[0])), list(chain(*pose0[0])))
            img = write_on_image(img=img, text=str(score), color=[0, 0, 0])
        except Exception as err:
            print('Error:', err)
            pass

        cv2.imshow('My Pose', img)

if __name__ == '__main__':
    main()
