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

**Office Pose** is the application that will help users to do exercises to reduce pain from Office Syndrome symptoms or even prevent them from getting one. The application will let users use their camera to detect their body while doing the exercises and then evaluate the correctness of each posture.

# Why is it interesting

- Nowadays, people spend most of their time sitting in front of a computer which eventually cause the Office Syndrome. Doing exercises can help prevent them from being suffer from the syndrome, but it is not easy to correctly do the exercise. Moreover, doing the exercise incorrectly might lead to other injuries. To solve the problem, the application were developed in order to help users do the exercise correctly and to let users do office syndrome exercises with more fun.
- Regarding the developer team, this project also leads to exploring existing frameworks as well as studying about pose keypoints extraction and comparison methods.

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
3. Put keypoint coordinates in `exercise.py` and define the weight of each keypoint to be used in the scoring step. Each keypoint specified in this file consist of 3 elements: x coordinate, y coordinate, and weight. Use zero as weight value if the keypoints must be neglected when evaluate the correctness, for example leg keypoints will not be considered for half-body exercise.
4. Use `flip_image.py` to flip the image if the exercise can be done from both left and right direction

## Get Inputs

1. The application get a user image as an input from the camera (image source is defined by `--video=<source>`)
2. The image is resize according to the resolution defined by `--resolution=<resolution>`
3. Then the image is passed to the model to predict the keypoints.

## Normalize Keypoints

1. All the keypoints are translated such that the nose keypoint becomes the origin of the coordinate system. This step is done by subtract nose keypoint coordinate values from every keypoint.
2. For x-axis scaling operation, all x keypoints are divide by the distance between left and right shoulder keypoint so that the distance between the left shoulder and right shoulder keypoints becomes 1. Regarding y-axis, all y keypoints are divided by the distance between shoulder and hip and then multiplied by 2 in order to preserve the normal human body ratio of the skeleton.
3. In case of side pose exercise, the keypoints are scaled such that the distance between the shoulder and hip keypoints becomes 1.

## Pose Compare (Scoring)

1. After the exercise keypoints and user pose keypoints were normalized, the two sets of normalized keypoints are compared using the weighted Euclidean distance. Each keypoint's weight were specified manually in the Prepare Exercises step.
2. Transform the euclidean distance value into out-of-ten score by setting the score to 10 if the distance is 0 and to 0 if the distance equal to the distance between exercise keypoints and all (0,0)'s keypoints.
3. The mapping is not linear to increase the range of score in case the distance is close.
4. If a user can do the pose correctly to the point that the score exceeds 7, the 5-second timer will start to countdown. The user will finish an exercise after maintaining the score to be more than 7 for 5 seconds.

## Integrate with GUI

1. The application uses TKinter framework to help building the GUI.
2. There is a single thread that will run OpenPifPaf backend parallelly to prevent application from blocking.

# Results

### The application consists of 2 modes

- **Normal mode** - do all exercises
- **Free Play mode** - do only one exercise

### Example of the application GUI

| GUI components                             | GUI images                                                                                                           |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Main page                                  | ![Main Page](https://github.com/WachirapatMT/Office_Pose/blob/main/gui_images/main%20page.jpg?raw=true)              |
| Exercise selection window (free play mode) | ![Free Play Modal](https://github.com/WachirapatMT/Office_Pose/blob/main/gui_images/free%20play%20mode.jpg?raw=true) |
| Exercise page                              | ![Exercise Page](https://github.com/WachirapatMT/Office_Pose/blob/main/gui_images/exercise%20page.jpg?raw=true)      |

### Demo

[![Office Pose Demo](https://img.youtube.com/vi/bO1u5aaBY-o/0.jpg)](https://youtu.be/bO1u5aaBY-o)

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

# Team Member

- 6131022721 Borvornwit Limpawitayakul
- 6131040021 Wachirapat Manorat
- 6131048021 Siratish Sakpiboonchit
- 6131314021 Techid Janphaka
