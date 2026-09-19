# Sample Mid-Term Examination (30 marks)

**MCTE 4323 / MCTA 4364 Machine Vision** - covers Weeks 1-7.
Time: 1 hour 30 minutes. Answer all questions. Calculators permitted.

---

## Section A - Short answers (10 marks, 2 each)

**A1.** Distinguish between **image processing**, **computer vision**, and **machine vision** with one example of each.

**A2.** A camera sensor integrates light over a fixed exposure time. Explain how this affects motion blur in a fast-moving conveyor inspection task.

**A3.** State the pinhole projection equations relating a 3D point $(X, Y, Z)$ to pixel coordinates $(u, v)$ using $f_x, f_y, c_x, c_y$.

**A4.** Explain the difference between the **Gaussian** and **Laplacian** image pyramids.

**A5.** Why is **CLAHE** often preferred over global histogram equalization for outdoor images?

---

## Section B - Calculations (10 marks, 5 each)

**B1.** A stereo camera has intrinsics $f_x = f_y = 800$ px, principal point $(320, 240)$, baseline $B = 0.10$ m. A feature is detected at left pixel $(400, 300)$ and right pixel $(360, 300)$.
   (a) Compute the disparity. (2)
   (b) Compute the depth $Z$. (2)
   (c) Compute the 3D coordinates $(X, Y)$ of the point. (1)

**B2.** A 3x3 convolution kernel is
$$K = \\frac{1}{9}\\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & 1 & 1 \\\\ 1 & 1 & 1 \\end{bmatrix}.$$
   (a) What is this filter called and what is its effect? (2)
   (b) A 3x3 image patch (grayscale) is
   $$\\begin{bmatrix} 10 & 10 & 10 \\\\ 10 & 100 & 10 \\\\ 10 & 10 & 10 \\end{bmatrix}.$$
   Compute the filter output at the centre pixel. (3)

---

## Section C - Design and application (10 marks)

**C1.** You must inspect metal washers on a production line and **count** them, rejecting bent ones. Propose a complete pipeline (from acquisition to decision), naming specific methods for each stage and justifying your choices. Include how you would evaluate the system. (10)

---

## Mark scheme

<details><summary>Show mark scheme</summary>

### Section A
- **A1:** Image processing transforms images (e.g. denoise); computer vision extracts information from images (e.g. object detection); machine vision uses vision for an engineering decision/action (e.g. reject a defective part). 1 mark each idea, examples necessary.
- **A2:** Longer exposure accumulates motion, so fast objects blur; reduce exposure time or add strobe/global-shutter lighting. 2 marks.
- **A3:** $u = f_x X/Z + c_x$, $v = f_y Y/Z + c_y$. 2 marks (1 each).
- **A4:** Gaussian pyramid is a smoothed, downsampled sequence; Laplacian pyramid stores the difference (high-frequency detail) between consecutive Gaussian levels. 2 marks.
- **A5:** CLAHE equalizes locally in tiles and clips contrast, avoiding global over-amplification and preserving local detail under varying illumination. 2 marks.

### Section B
- **B1:**
  - (a) $d = 400 - 360 = 40$ px. (2)
  - (b) $Z = fB/d = 800 \\times 0.10 / 40 = 2.0$ m. (2)
  - (c) $X = (400-320)\\times 2/800 = 0.20$ m; $Y = (300-240)\\times 2/800 = 0.15$ m. (1)
- **B2:**
  - (a) 3x3 **box/mean** filter; averages the neighbourhood to blur/smooth the image. (2)
  - (b) Mean of the nine values = $(10\\times8 + 100)/9 = 180/9 = 20$. (3)

### Section C (indicative bands)
- Acquisition and lighting setup appropriate to metal (e.g. diffuse/bright-field lighting to reveal shape). (2)
- Segmentation (thresholding + morphology, or contour-based) and counting via connected components. (3)
- Bent-washer rejection via shape features (circularity, aspect ratio, Hu moments) or a small classifier. (3)
- Evaluation: precision/recall of count, false accept/reject rate, comparison to manual ground truth. (2)

</details>
