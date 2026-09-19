# Week 04 - Data Structures for Image Analysis
> **Official syllabus block:** Matrices, chains, topological and relational data structures, pyramids, quadtrees (Chapter 4).

## Learning objectives
- Represent images and regions using suitable data structures.
- Build Gaussian and Laplacian pyramids.
- Use contours and their hierarchy.
- Implement quadtree decomposition.

## Key concepts
- Pyramids provide multi-scale representations.
- Contours and chain codes represent boundaries.
- Hierarchy encodes containment (holes and nested objects).
- Quadtrees adapt resolution to image content.

## Key formulas
- Each pyramid level halves width and height (quarter of pixels).
- Laplacian level = Gaussian level - upsampled next level.

## Common misconceptions
- More pyramid levels do not add information; they discard detail.
- Contour area is not the same as the number of boundary pixels.
- Quadtree block size must be a power of two.

## In the lab
Open `notebooks/04_image_structures/04_image_structures.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV contours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- OpenCV pyramids: https://docs.opencv.org/4.x/d4/d1f/tutorial_pyramids.html
