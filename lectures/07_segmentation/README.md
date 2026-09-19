# Week 07 - Image Segmentation
> **Official syllabus block:** Thresholding, edge-based segmentation, region-based segmentation, matching, evaluation issues, texture (Chapter 6).

## Learning objectives
- Segment images by thresholding.
- Detect edges and parametric shapes.
- Segment by regions and separate touching objects.
- Evaluate segmentation quality.

## Key concepts
- Global, Otsu and adaptive thresholding.
- Canny edge detection and Hough transforms.
- Connected components, flood fill, watershed.
- Texture-based segmentation.

## Key formulas
- Otsu maximises between-class variance.
- Hough lines: rho = x cos(theta) + y sin(theta).
- Gradient magnitude for edges: sqrt(Gx^2 + Gy^2).

## Common misconceptions
- Otsu assumes a bimodal histogram; it fails on uneven lighting.
- Canny thresholds are not IoU-style; they control hysteresis.
- Watershed over-segments without good markers.

## In the lab
Open `notebooks/07_segmentation/07_segmentation.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV thresholding: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- OpenCV watershed: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
