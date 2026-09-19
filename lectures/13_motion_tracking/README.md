# Week 13 - Dynamic Vision
> **Official syllabus block:** Change detection, segmentation using motion, motion correspondence, image flow, segmentation using a moving camera, tracking, shape from motion, case studies (Chapter 16).

## Learning objectives
- Detect motion.
- Compute sparse and dense optical flow.
- Track multiple objects.
- Count events such as line crossings.

## Key concepts
- Frame differencing and background subtraction.
- Lucas-Kanade and Farneback flow.
- Association and Kalman filtering.
- ByteTrack and DeepSORT.

## Key formulas
- LK assumes small motion and brightness constancy.
- Flow magnitude approximates apparent speed.

## Common misconceptions
- Optical flow is apparent motion, not true motion.
- Frame differencing misses stopped objects.
- Centroid tracking suffers identity switches on occlusion.

## In the lab
Open `notebooks/13_motion_tracking/13_motion_tracking.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV optical flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- Ultralytics track: https://docs.ultralytics.com/modes/track/
