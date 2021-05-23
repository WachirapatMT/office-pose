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
(Te)
what

# Why is it interesting
(Te)
why

# Technical Challenges

### The definition of the correctness of an exercise

It might be defined using geometric-heuristic such as detecting the angle of arms and legs, or maybe using euclidean distance to evaluate the difference between a user pose and an correct example pose.

### Users with different body ratio

A user pose will be compared with a correct exercise. However, different users have different body ratio which might affect the comparison outcome.

### Users' different camera aspect

Different camera aspect provides different view of an image which may affect the pose comparision, but it cannot be controlled since the camera is set by a user.

### Real-time application

The FPS must not be too low and the correctness score should be calculated and showed to a user immediately.

# Related Works

### [Pose Trainer](https://deepai.org/publication/pose-trainer-correcting-exercise-posture-using-pose-estimation)

Pose Trainer is an application that detects the user's exercise pose to help correct the pose. The application use OpenPose the detect the keypoints then evaluate the pose using geometric-heuristic and machine learning algorithms.

### [Physio Pose](https://medium.com/@_samkitjain/physio-pose-a-virtual-physiotherapy-assistant-7d1c17db3159)

Physio Pose is an application that help people do exercises by provide instant feedback and act as a personal virtual trainer. The application use Openpifpaf as a pose estimation model and use geometric-heuristic for evaluation.

# Methodology

## Select Model

[Openpifpaf](https://openpifpaf.github.io/intro.html) were chosen to be a pose estimation model since it is both CPU and GPU supported. Moreover, the accuracy were good even for low resolution images and half body images.

## Prepare Exercises

1. Choose an exercise image from the internet
2. Use `visualise_image.py` to detect keypoints from the image
3. Put keypoint coordinates in `exercise.py` and define the weight of each keypoint to be used in the scoring step.
4. Use `flip_image.py` to flip the image if the exercise can be done from both left and right direction

## Get Inputs

1. The application get a user image as an input from the camera (image source is defined by `--video=<source>`)
2. The image is resize according to the resolution defined by `--resolution=<resolution>`
3. Then the image is passed to the model to predict the keypoints.

## Normalize Keypoints

1. All the keypoints are translated such that the nose keypoint becomes the origin of the coordinate system.
2. The keypoints are scaled such that the distance between the left shoulder and right shoulder keypoints becomes 1.
3. In case of side pose exercise, the keypoints are scaled such that the distance between the shoulder and hip keypoints becomes 1.

## Pose Compare (Scoring)

1. The two sets of normalized keypoints are compared using the weighted Euclidean distance.
2. Map the distance into out-of-ten score by setting the score to 10 if the distance is 0 and to 0 if the distance equal to the distance between exercise keypoints and all (0,0)'s keypoints.
3. The mapping is not linear to increase the range of score in case the distance is close.

## Integrate with GUI
(Te)

# Results

### The application consists of 2 modes

- **Normal mode** - do all exercises
- **Free Play mode** - do only one exercise

### Example of the application GUI
|  GUI components   | GUI images |
|-------------------|-----------|
| Main page |![Main Page](https://github.com/WachirapatMT/Office_Pose/blob/main/gui_images/main%20page.jpg?raw=true)|
|Exercise selection window (free play mode)|![Free Play Modal](https://github.com/WachirapatMT/Office_Pose/blob/main/gui_images/free%20play%20mode.jpg?raw=true)|
Exercise page|![Exercise Page](https://github.com/WachirapatMT/Office_Pose/blob/main/gui_images/exercise%20page.jpg?raw=true)|

# Limitations

### Processing power

Since the application were developed using CPU, it cannot be improved to the full extent.

### Model accuracy

According to the processing power limitation, the complicated model cannot be employed and the chosen model were configured to use less resources, which provides lower accuracy, so that application the can be run smoothly.

### Testing environment

The application required a user to do exercises in a wide space where the camera can capture the whole body. Such space was not available for every user, so it was difficult to control the testing environment of different testers.

# Recommendations for Future Improvement

- **Counting execise reps** since some exercises might need to be done more than one time.
- **Specifying bad keypoints** so that users know which parts of the pose are needed to be corrected.
- **Handling image rotation by using linear tranformation** since users' camera might not be set straight which affects the exercise score.
- **Handling undetected keypoins using 3D pose estimation** so that exercise image with missing body parts such as hand behind the head can be used in the application.
