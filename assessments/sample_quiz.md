# Sample Quiz (10 marks)

**MCTE 4323 / MCTA 4364 Machine Vision** - revision quiz. Time: 20 minutes.

Instructions: answer all questions. One mark each.

1. An 8-bit colour image has shape `(480, 640, 3)`. How much memory does the raw pixel data occupy (in bytes)?
2. In OpenCV, what colour-channel order does `cv2.imread` return by default?
3. State **one** advantage of HSV over BGR for colour segmentation.
4. Define **sampling** and **quantization** in the context of image digitisation.
5. What does the **principal point** in the camera intrinsic matrix represent?
6. A stereo rig has focal length $f = 500$ px and baseline $B = 0.12$ m. Compute the depth of a point with disparity $d = 20$ px.
7. State the **Otsu** thresholding assumption that limits its use under uneven lighting.
8. What two thresholds does the Canny edge detector use, and what is their purpose?
9. Give one reason a **median** filter is preferred over a Gaussian filter.
10. What does `cv2.findHomography(..., cv2.RANSAC, ...)` return besides the matrix?

---

## Answers

<details><summary>Show answers</summary>

1. `480 x 640 x 3 = 921,600` bytes (about 0.92 MB).
2. **BGR** (blue, green, red).
3. Hue separates colour from brightness, so segmentation is more robust to lighting changes.
4. **Sampling** = discretising spatial position into a pixel grid. **Quantization** = discretising intensity into a finite number of levels.
5. The image coordinates $(c_x, c_y)$ where the optical axis intersects the image plane.
6. $Z = fB/d = 500 \\times 0.12 / 20 = 3.0$ m.
7. It assumes a **bimodal** histogram (two clear intensity classes) that is consistent across the image.
8. A low threshold and a high threshold (hysteresis): weak edges are kept only if connected to strong edges.
9. Median filtering removes salt-and-pepper (impulse) noise while preserving edges; Gaussian blur softens edges.
10. A **mask** that flags inlier matches used to fit the homography (and it uses RANSAC to reject outliers).

</details>
