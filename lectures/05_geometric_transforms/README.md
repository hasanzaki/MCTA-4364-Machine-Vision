# Week 05 - Preprocessing I: Brightness and Geometric Transforms
> **Official syllabus block:** Pixel brightness transformation and geometric transformation (Chapter 5).

## Learning objectives
- Apply brightness, contrast and gamma transformations.
- Improve contrast with histogram equalization and CLAHE.
- Apply affine transformations.
- Apply perspective transformation and homography.

## Key concepts
- Point operations use only the pixel value.
- Histogram equalization redistributes intensities; CLAHE is local.
- Affine preserves parallel lines; perspective does not.
- Homography maps one plane to another.

## Key formulas
- g = alpha * f + beta (contrast and brightness).
- Gamma: g = 255 * (f / 255)^(1/gamma).
- Homography has 8 degrees of freedom.

## Common misconceptions
- Gamma is not a linear matrix operation on pixel values.
- Global equalization can amplify noise; CLAHE limits this.
- An affine transform cannot rectify a tilted document.

## In the lab
Open `notebooks/05_geometric_transforms/05_geometric_transforms.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV geometric transformations: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- OpenCV histogram equalization: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
