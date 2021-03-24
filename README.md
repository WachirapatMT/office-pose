## exercise_compare.py

Compare user's pose and the specified exercise pose

**Important Arguments**

- `compare` - turn on compare mode
- `video` - define video source {0: default webcam, 1: extra webcam, or path to a video}
- `skeleton` - show skeleton model
- `joint` - show keypoints
- `exercise` - specified an exercise to be compared
- `resolution` - just a resolution value

#

## visualise_image.py

Calculate and visualise keypoints of a single image

**Important Arguments**

- `override` - use hard code keypoints instead of returned keypoints
- `resolution` - just a resolution value


#

## common.py

Store common functions to be imported by other modules

#

## flip_image.py

Apply horizontal flip to a specified image

#

## keypoints format

|              | keypoints    |                 |
| ------------ | ------------ | --------------- |
| nose.x       | nose.y       | nose.prob       |
| l.eye.x      | l.eye.y      | l.eye.prob      |
| r.eye.x      | r.eye.y      | r.eye.prob      |
| l.ear.x      | l.ear.y      | l.ear.prob      |
| r.ear.x      | r.ear.y      | r.ear.prob      |
| l.shoulder.x | l.shoulder.y | l.shoulder.prob |
| r.shoulder.x | r.shoulder.y | r.shoulder.prob |
| l.elbow.x    | l.elbow.y    | l.elbow.prob    |
| r.elbow.x    | r.elbow.y    | r.elbow.prob    |
| l.wrist.x    | l.wrist.y    | l.wrist.prob    |
| r.wrist.x    | r.wrist.y    | r.wrist.prob    |
| l.hip.x      | l.hip.y      | l.hip.prob      |
| r.hip.x      | r.hip.y      | r.hip.prob      |
| l.knee.x     | l.knee.y     | l.knee.prob     |
| r.knee.x     | r.knee.y     | r.knee.prob     |
| l.ankle.x    | l.ankle.y    | l.ankle.prob    |
| r.ankle.x    | r.ankle.y    | r.ankle.prob    |
