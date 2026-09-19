# Week 12 - 3D Vision
> **Official syllabus block:** Projective geometry, scene reconstruction from multiple views, stereo vision, trifocal sensor, 3D information from radiometric measurements, case studies (Chapter 11).

## Learning objectives
- Explain depth from stereo.
- Compute disparity and depth.
- Back-project to a point cloud.
- Visualise and interpret point clouds.

## Key concepts
- Epipolar geometry and rectification.
- Block matching and SGBM.
- Intrinsics for back-projection.
- Point-cloud processing and registration.

## Key formulas
- Z = f B / d.
- X = (u - cx) Z / f; Y = (v - cy) Z / f.
- Depth error grows with Z^2 / (f B).

## Common misconceptions
- A point cloud needs calibration to be metric.
- Textureless regions give unreliable disparity.
- More points is not always better; density and accuracy differ.

## In the lab
Open `notebooks/12_3d_point_clouds/12_3d_point_clouds.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV depth maps: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- Open3D tutorials: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
