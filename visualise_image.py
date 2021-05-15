import argparse
import base64

import openpifpaf
import torch
import cv2

from common import (
    visualise,
)
from processor import Processor


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
        default=0.6,
        type=float,
        help=(
            "Resolution prescale factor from 640x480."
            "Will be rounded to multiples of 16."
        ),
    )
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Override keypoints with hard code values",
    )

    args = parser.parse_args()

    # Add args.device
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


def image_pose(image_path="exercise_images/look_up_right.png"):
    args = cli()

    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    width_height = (
        int(width * args.resolution // 16) * 16,
        int(height * args.resolution // 16) * 16,
    )

    processor_singleton = Processor(width_height, args)
    keypoint_sets, scores, width_height = processor_singleton.single_image(
        b64image=base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("UTF-8")
    )
    print("keypoint:::", [keypoint_sets[0].tolist()])

    # Use hard coding keypoints to visualise or adjust the keypoints
    if args.override:
        keypoint_sets = [
            [
                [4.7093248e-01, 2.0426853e-01, 8.6702675e-01],
                [4.8350826e-01, 1.9285889e-01, 9.0909678e-01],
                [4.5933738e-01, 1.9329509e-01, 7.9952747e-01],
                [5.0037909e-01, 2.1018410e-01, 6.8106771e-01],
                [4.4295478e-01, 2.0732497e-01, 8.4605628e-01],
                [5.1420957e-01, 2.8525087e-01, 9.5134568e-01],
                [4.2685834e-01, 2.8133684e-01, 1.2184765e00],
                [5.2738458e-01, 1.5919007e-01, 8.1505233e-01],
                [4.1852579e-01, 1.7545088e-01, 7.9655707e-01],
                [4.8712268e-01, 8.4518030e-02, 7.3338878e-01],
                [4.5714667e-01, 8.1567958e-02, 1.0000000e-03],
                [5.0157762e-01, 5.1679271e-01, 5.8042961e-01],
                [4.4701675e-01, 4.9484652e-01, 7.6062948e-01],
                [6.2654567e-01, 6.2967056e-01, 4.0503103e-01],
                [4.7448894e-01, 7.3578298e-01, 3.6731312e-01],
                [4.9454674e-01, 7.3082601e-01, 1.0000000e-03],
                [4.9181211e-01, 8.8942599e-01, 4.6938220e-01],
            ]
        ]

    img = visualise(
        img=img,
        keypoint_sets=keypoint_sets,
        width=width,
        height=height,
        vis_keypoints=True,
        vis_skeleton=True,
    )

    cv2.imshow("Image Keypoints", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_pose()