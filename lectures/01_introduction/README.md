# Week 01 - Introduction to Machine Vision
> **Official syllabus block:** Machine vision, image geometry, sampling and quantization, image definitions, levels of computation, road map (Chapter 1).

## Learning objectives
- Define machine vision and distinguish it from computer vision and image processing.
- Describe the imaging pipeline from light to a digital image.
- Explain sampling and quantization and their effect on image quality.
- Set up a Python/Colab environment for the course.

## Key concepts
- Machine vision = sensing + processing + decision for engineering tasks.
- Pipeline: light -> optics -> sensor -> digitization -> array (H, W, C).
- Sampling = spatial discretisation; quantization = intensity discretisation.
- Levels of computation: low (pixels), mid (features), high (interpretation).

## Key formulas
- Spatial resolution = 1 / sampling interval; Nyquist limit applies.
- Number of grey levels = 2^(bit depth).
- Memory = H x W x C bytes for 8-bit images.

## Common misconceptions
- Machine vision is not only deep learning; classical methods remain essential.
- A higher-resolution image is not always better - it costs memory and compute.
- Colour is not always needed; many industrial tasks use grayscale.

## In the lab
Open `notebooks/01_introduction/01_introduction.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV Python tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- Wikipedia - Computer vision: https://en.wikipedia.org/wiki/Computer_vision
