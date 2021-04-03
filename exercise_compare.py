import argparse
import base64
import csv
import time
import os

from typing import List
from itertools import chain

import cv2
import numpy as np
import openpifpaf
import torch
from scipy.spatial.distance import euclidean

from common import CocoPart, SKELETON_CONNECTIONS, write_on_image, visualise, normalise
from processor import Processor
from exercise import EXERCISE


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    openpifpaf.decoder.cli(
        parser, force_complete_pose=True, instance_threshold=0.2, seed_threshold=0.5
    )
    openpifpaf.network.nets.cli(parser)
    parser.add_argument(
        "--resolution",
        default=0.4,
        type=float,
        help=(
            "Resolution prescale factor from 640x480. "
            "Will be rounded to multiples of 16."
        ),
    )
    parser.add_argument("--video", default=None, type=str, help="Video source")
    parser.add_argument(
        "--exercise",
        default="side_bend_left",
        type=str,
        help="Exercise ID to perform.",
    )

    vis_args = parser.add_argument_group("Visualisation")
    vis_args.add_argument(
        "--joints",
        default=False,
        action="store_true",
        help="Draw joint's keypoints on the output video.",
    )
    vis_args.add_argument(
        "--skeleton",
        default=False,
        action="store_true",
        help="Draw skeleton on the output video.",
    )
    vis_args.add_argument(
        "--compare",
        default=False,
        action="store_true",
        help="Compare between user pose and the chosen exercise",
    )

    # vis_args.add_argument('--save-output', default=False, action='store_true',
    #                       help='Save the result in a video file.')
    # vis_args.add_argument('--fps', default=25, type=int,
    #                       help='FPS for the output video.')
    # vis_args.add_argument('--out-path', default='pose_result.mp4', type=str,
    #                       help='Save the output video at the path specified. .avi file format.')

    args = parser.parse_args()

    # Add args.device
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


def main():
    score = 10
    task_finish = 0
    args = cli()

    # Get exercise keypoints
    exercise = EXERCISE[args.exercise]
    exercise_img = cv2.imread(os.path.join("exercise_images", f"{args.exercise}.png"))

    # Video source
    if args.video is None:
        print("Video source: default webcam")
        cam = cv2.VideoCapture(0)
    elif args.video == "1":
        print("Video source: mobile camera")
        cam = cv2.VideoCapture(1)
    else:
        print(f"Video source: {args.video}")
        cam = cv2.VideoCapture(args.video)

    ret_val, img = cam.read()
    height, width = img.shape[:2]
    exercise_img = cv2.resize(exercise_img, img.shape[:2][::-1])

    # Resize image to multiple of 16 due to some unknown convention
    width_height = (
        int(width * args.resolution // 16) * 16,
        int(height * args.resolution // 16) * 16,
    )
    print(f"Resize image from {(width, height)} to {width_height}")

    # Initialise model
    processor_singleton = Processor(width_height, args)

    # For FPS calculation
    start = time.time()
    frame = 0

    while True:
        frame += 1

        ret_val, img = cam.read()

        if not ret_val or img is None:
            continue

        # Press `esc` to exit
        if task_finish == 3 or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            end = time.time()
            print(f"FPS: {frame/(end-start)}")
            break

        keypoint_sets, scores, width_height = processor_singleton.single_image(
            b64image=base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("UTF-8")
        )

        ### Normal Mode ###
        if not args.compare:
            img = visualise(
                img=img,
                keypoint_sets=keypoint_sets,
                width=width,
                height=height,
                vis_keypoints=args.joints,
                vis_skeleton=args.skeleton,
            )

        try:
            # map to a length of 2 before passing to normalise()
            my_pose = [list(map(lambda x: [x[0], x[1]], keypoint_sets[0]))]
            exercise_pose = [list(map(lambda x: [x[0], x[1]], exercise[0]))]

            my_pose_norm = normalise(my_pose)
            exercise_pose_norm = normalise(exercise_pose)

            ### Compare Mode ###
            if args.compare:
                # comment out to use real image as a background
                img = np.ones((480, 640, 3), np.uint8) * 255

                img = visualise(
                    img=img,
                    keypoint_sets=my_pose_norm,
                    width=width // 12,
                    height=height // 12,
                    tranX=400,
                    tranY=110,
                    vis_keypoints=args.joints,
                    vis_skeleton=True,
                )
                img = visualise(
                    img=img,
                    keypoint_sets=exercise_pose_norm,
                    width=width // 12,
                    height=height // 12,
                    tranX=150,
                    tranY=110,
                    vis_keypoints=args.joints,
                    vis_skeleton=True,
                )
            ###############
            
            weight_list = list(map(lambda x: [1,1] if x[2] >= 1e-5 else [0,0],keypoint_sets[0]))
            normalize_weight = list(chain(*weight_list)).count(1)
            print(normalize_weight)
            # Show similarity score on image
            #score with no ommit keypoint
            # score = euclidean(list(chain(*my_pose[0])), list(chain(*exercise_pose[0]))) 
            #score with ommited keypoint
            score = euclidean(list(chain(*my_pose[0])), list(chain(*exercise_pose[0])),list(chain(*weight_list)))/ (normalize_weight if normalize_weight!=0 else 1)
            print("test")
            # if score < 0.25:
            #     task_finish += 1
            #     if task_finish == 3:
            #         score = "Correct"
            # else:
            #     task_finish = 0

        except Exception as err:
            print("Error:", err)
            pass

        # Add image to the side
        img = np.hstack((exercise_img, img))

        img = write_on_image(img=img, text=f"{task_finish} - {score}", color=[0, 0, 0])
        cv2.imshow("My Pose", img)


if __name__ == "__main__":
    main()
