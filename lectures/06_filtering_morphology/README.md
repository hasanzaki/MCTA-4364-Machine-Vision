# Week 06 - Preprocessing II: Filtering and Morphology
> **Official syllabus block:** Local preprocessing and image restoration (Chapter 5).

## Learning objectives
- Apply convolution with custom and standard kernels.
- Compare linear and non-linear filters.
- Restore images corrupted by noise.
- Use morphology to clean binary shapes.

## Key concepts
- Convolution and correlation with kernels.
- Mean and Gaussian smoothing; median and bilateral filtering.
- Derivative filters (Sobel, Laplacian) and sharpening.
- Morphology: erosion, dilation, opening, closing, gradient.

## Key formulas
- Filter response is a weighted sum over a neighbourhood.
- Opening = erosion then dilation; closing = dilation then erosion.

## Common misconceptions
- A larger kernel is not always better; it removes detail.
- Gaussian blur is poor for salt-and-pepper noise - use median.
- Morphology requires a binary or grayscale image, not colour.

## In the lab
Open `notebooks/06_filtering_morphology/06_filtering_morphology.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV smoothing: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- OpenCV morphology: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
