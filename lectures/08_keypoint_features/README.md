# Week 08 - Keypoint Features and Matching
> **Official syllabus block:** Object recognition: region identification, contour and region shape representation, knowledge representation, graph matching, optimization techniques (Chapter 9).

## Learning objectives
- Detect keypoints with Harris, Shi-Tomasi, SIFT and ORB.
- Compute and match descriptors.
- Apply the ratio test and RANSAC.
- Estimate homography for stitching or pose.

## Key concepts
- Corners are distinctive in two directions.
- SIFT = float 128-D descriptor; ORB = binary, fast.
- Matching with BF/FLANN plus the ratio test.
- Robust fitting with RANSAC.

## Key formulas
- Harris score: R = det(M) - k trace(M)^2.
- Ratio test: accept match if d1 < ratio * d2.

## Common misconceptions
- More keypoints is not always better; redundant and noisy points hurt.
- The ratio test reduces but does not remove outliers.
- Homography is valid only for planar scenes or pure rotation.

## In the lab
Open `notebooks/08_keypoint_features/08_keypoint_features.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV feature detection: https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
- Wikipedia - SIFT: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
