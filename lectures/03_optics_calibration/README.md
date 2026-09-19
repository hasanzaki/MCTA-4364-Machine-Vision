# Week 03 - Image Formation, Optics and Calibration
> **Official syllabus block:** Radiometric measurement, capture and optics, lens aberration and distortion, radiometric image capture, surface reflectance (Chapter 3).

## Learning objectives
- Apply the pinhole camera model to project 3D points to pixels.
- Explain intrinsic and extrinsic parameters.
- Describe radial and tangential lens distortion.
- Recover camera parameters by calibration.

## Key concepts
- Pinhole model maps (X, Y, Z) to (u, v) via K, R, t.
- Intrinsics K: focal length in pixels and principal point.
- Extrinsics [R | t]: camera pose in the world.
- Distortion coefficients model lens imperfection.

## Key formulas
- u = fx * X / Z + cx, v = fy * Y / Z + cy.
- Radial distortion: x' = x(1 + k1 r^2 + k2 r^4).
- k1 < 0 barrel, k1 > 0 pincushion.

## Common misconceptions
- Calibration is not a one-time setup; it depends on zoom and focus.
- Do not confuse focal length in millimetres with focal length in pixels.
- Undistortion changes pixel coordinates, which matters for measurement.

## In the lab
Open `notebooks/03_optics_calibration/03_optics_calibration.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV camera calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Wikipedia - Pinhole camera model: https://en.wikipedia.org/wiki/Pinhole_camera_model
