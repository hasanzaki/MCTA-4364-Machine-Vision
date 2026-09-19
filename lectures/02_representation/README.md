# Week 02 - Image Representation and Colour
> **Official syllabus block:** Digitization, digital image properties, colour images, cameras (Chapter 2).

## Learning objectives
- Represent an image as an array and inspect its properties.
- Convert between BGR, RGB, HSV and Lab colour spaces.
- Read and interpret an intensity histogram.
- Segment objects by colour.

## Key concepts
- Image properties: shape, dtype, resolution, contrast, entropy.
- Colour spaces organise the three channels differently for different tasks.
- HSV separates hue from brightness, aiding segmentation.
- Cameras: sensor size, pixel size, dynamic range, noise.

## Key formulas
- Quantization to B bits gives 2^B levels.
- Contrast measures the spread of intensities.
- Entropy measures the average information per pixel.

## Common misconceptions
- OpenCV uses BGR, not RGB - a frequent display bug.
- Hue is circular, so red wraps from 179 to 0.
- A colour can be invisible in grayscale but obvious in colour.

## In the lab
Open `notebooks/02_representation/02_representation.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- OpenCV colour conversions: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
- Wikipedia - HSL and HSV: https://en.wikipedia.org/wiki/HSL_and_HSV
