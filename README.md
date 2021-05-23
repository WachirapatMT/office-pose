# Instruction
1. Clone project
2. Install required packages defined in `requirements.txt`
3. Run `python gui.py`
4. Some arguments can be defined in the run script
    - `--compare` - turn on comparison mode
    - `--video=<source>` - define video source {0: default webcam, 1: extra webcam, or path to a video}
    - `--skeleton` - show skeleton model
    - `--joint` - show keypoints
    - `--resolution=<resolution>` - define resolution value in range 0-1 (default: 0.4)
# Project Description

what

# Why is it interesting

why

# Technical Challenges

## The definition of the correctness of an exercise
It might be defined using geometric-heuristic such as detecting the angle of arms and legs, or maybe using euclidean distance to evaluate the difference between a user pose and an correct example pose. 
## Users with different body ratio
A user pose will be compared with a correct exercise. However, different users have different body ratio which might affect the comparison outcome. 
## Users' different camera aspect
Different camera aspect provides different view of an image which may affect the pose comparision, but it cannot be controlled since the camera is set by a user.
## Real-time application
The FPS must not be too low and the correctness score should be calculated and showed to a user immediately.
# Related Works

## [Pose Trainer](https://deepai.org/publication/pose-trainer-correcting-exercise-posture-using-pose-estimation)

Pose Trainer is an application that detects the user's exercise pose to help correct the pose. The application use OpenPose the detect the keypoints then evaluate the pose using geometric-heuristic and machine learning algorithms.

## [Physio Pose](https://medium.com/@_samkitjain/physio-pose-a-virtual-physiotherapy-assistant-7d1c17db3159)

Physio Pose is an application that help people do exercises by provide instant feedback and act as a personal virtual trainer. The application use Openpifpaf as a pose estimation model and use geometric-heuristic for evaluation.

# Methodology

## Select Model

## Prepare Exercises

## Get Inputs

### Images

### Argument



## Normalize Keypoints

## Pose Compare (Scoring)

## Integrate with GUI

# Results

may be some images

# Limitations

## Processing power

Since the application were developed using CPU, it cannot be improved to the full extent.

## Model accuracy

According to the processing power limitation, the complicated model cannot be employed and the chosen model were configured to use less resources, which provides lower accuracy, so that application the can be run smoothly.

## Testing environment
The application required a user to do exercises in a wide space where the camera can capture the whole body. Such space was not available for every user, so it was difficult to control the testing environment of different testers.
# Recommendations for Future Improvement
- **Counting execise reps** since some exercises might need to be done more than one time.
- **Specifying bad keypoints** so that users know which parts of the pose are needed to be corrected.
- **Handling image rotation by using linear tranformation** since users' camera might not be set straight which affects the exercise score.
- **Handling undetected keypoins using 3D pose estimation** so that exercise image with missing body parts such as hand behind the head can be used in the application.
