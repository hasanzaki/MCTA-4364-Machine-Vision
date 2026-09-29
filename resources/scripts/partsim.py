"""Procedural images of a machined flange on a conveyor, for inspection exercises.

The course uses the same simulated part in several weeks (integration lab, data-centric
machine vision, anomaly detection), so students meet a consistent "production line".
Everything is generated from a seed: no downloads, fully reproducible.

Nominal part (all sizes in mm):
    outer diameter 40.0 with an orientation notch (3.0 wide, 2.0 deep)
    centre hole 12.0, four bolt holes 4.0 on a 28.0 pitch-circle diameter
Camera: 0.2 mm per pixel at the default 256 x 256 resolution.

Defect classes: "ok", "missing_hole", "scratch", "chip", "undersize".
Extra defect for anomaly detection: "stain" (discoloured patch).
"""
import cv2
import numpy as np

CLASSES = ["ok", "missing_hole", "scratch", "chip", "undersize"]
SPEC = dict(outer_d=40.0, centre_d=12.0, bolt_d=4.0, pcd=28.0, notch_w=3.0, notch_depth=2.0)
MM_PER_PX = 0.2
UNDERSIZE_D = 38.6          # an undersize part is 1.4 mm (3.5%) too small; tolerance is +/- 0.4 mm
_SS = 4                     # supersampling factor for anti-aliased edges


def _brushed_texture(rng, h, w, angle_deg, strength):
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    k = np.zeros((15, 15), np.float32)
    k[7, :] = 1 / 15
    rot = cv2.getRotationMatrix2D((7, 7), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (15, 15))
    k /= k.sum() + 1e-6
    return cv2.filter2D(noise, -1, k) * strength


def make_part(rng=None, defect="ok", size=256, mm_per_px=MM_PER_PX, lighting=True, pose=True, noise=3.0, exposure=1.0):
    """Return (bgr_uint8_image, info) for one part.

    info = dict(defect, centre (x, y) px, angle_deg, mm_per_px, outer_d_mm, defect_box (x0, y0, x1, y1) or None)
    `pose=False` places the part centred with the notch pointing up (angle 0), handy for golden templates.
    `exposure` scales the brightness and `noise` is the sensor noise standard deviation (grey levels), to simulate
    a change of lighting or camera.
    """
    rng = np.random.default_rng(rng)
    S = size * _SS
    px = _SS / mm_per_px                       # supersampled pixels per mm
    cx = size / 2 + (rng.uniform(-8, 8) if pose else 0)
    cy = size / 2 + (rng.uniform(-8, 8) if pose else 0)
    angle = rng.uniform(0, 360) if pose else 0.0
    outer_d = UNDERSIZE_D if defect == "undersize" else SPEC["outer_d"] + rng.normal(0, 0.08)
    C = np.array([cx, cy]) * _SS
    th = np.deg2rad(angle)
    up = np.array([np.sin(th), -np.cos(th)])            # direction of the notch
    right = np.array([np.cos(th), np.sin(th)])

    mask = np.zeros((S, S), np.uint8)
    R = outer_d / 2 * px
    cv2.circle(mask, tuple(np.round(C).astype(int)), int(round(R)), 255, -1, cv2.LINE_8)
    # orientation notch: a rectangle cut into the rim
    nw, nd = SPEC["notch_w"] / 2 * px, SPEC["notch_depth"] * px
    corners = [C + up * (R + 5) + right * nw, C + up * (R + 5) - right * nw, C + up * (R - nd) - right * nw, C + up * (R - nd) + right * nw]
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 0)
    cv2.circle(mask, tuple(np.round(C).astype(int)), int(round(SPEC["centre_d"] / 2 * px)), 0, -1)
    holes = list(range(4))
    if defect == "missing_hole":
        holes.remove(int(rng.integers(0, 4)))
    for k in holes:
        a = th + np.pi / 4 + k * np.pi / 2
        p = C + SPEC["pcd"] / 2 * px * np.array([np.sin(a), -np.cos(a)])
        cv2.circle(mask, tuple(np.round(p).astype(int)), int(round(SPEC["bolt_d"] / 2 * px)), 0, -1)
    box = None
    if defect == "chip":
        a = th + rng.uniform(0.6, 2 * np.pi - 0.6)          # anywhere on the rim except near the notch
        rim = C + R * np.array([np.sin(a), -np.cos(a)])
        r_chip = rng.uniform(1.4, 2.2) * px
        pts = rim + rng.normal(0, 0.25 * r_chip, (7, 2)) + r_chip * np.stack([np.cos(np.linspace(0, 2 * np.pi, 7, endpoint=False)),
                                                                             np.sin(np.linspace(0, 2 * np.pi, 7, endpoint=False))], 1)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 0)
        box = np.r_[rim - r_chip * 1.3, rim + r_chip * 1.3] / _SS
    alpha = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    # surface appearance
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    metal = 165 + 12 * np.cos(rr / 9.0) * 0.3 + _brushed_texture(rng, size, size, rng.uniform(0, 180), 14)
    metal = metal - 0.08 * (rr - 50)                   # slight radial falloff (turned surface)
    belt = 48 + _brushed_texture(rng, size, size, 0, 6) + rng.normal(0, 2, (size, size))
    if defect == "scratch":
        for _ in range(50):                            # resample until the scratch lies (mostly) on metal
            smask = np.zeros((S, S), np.uint8)
            ring_r = rng.uniform(SPEC["centre_d"] / 2 + 1.5, SPEC["outer_d"] / 2 - 2.0) * px
            a0 = rng.uniform(0, 2 * np.pi)
            p0 = C + ring_r * np.array([np.cos(a0), np.sin(a0)])
            direction = rng.uniform(0, np.pi)
            length = rng.uniform(5, 10) * px
            d = np.array([np.cos(direction), np.sin(direction)])
            p1, p2 = p0 - d * length / 2, p0 + d * length / 2
            cv2.line(smask, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, int(rng.uniform(1.2, 2.0) * _SS))
            s = cv2.resize(smask, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if (s * alpha).sum() >= 0.85 * s.sum():
                break
        metal = metal - rng.uniform(45, 80) * s
        ys, xs = np.nonzero(s > 0.2)
        if len(xs):
            box = np.array([xs.min() - 2, ys.min() - 2, xs.max() + 2, ys.max() + 2], float)
    if defect == "stain":
        sx, sy = cx + rng.uniform(-60, 60), cy + rng.uniform(-60, 60)
        blob = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * rng.uniform(6, 11) ** 2))
        metal = metal - 45 * blob
        box = np.array([sx - 15, sy - 15, sx + 15, sy + 15])
    gray = alpha * metal + (1 - alpha) * belt
    img = np.stack([gray * 0.97, gray * 1.0, gray * 1.03], -1)            # slightly cool metal/LED tint
    if defect == "stain":
        img[..., 2] += 25 * blob * alpha                                     # brownish (more red, less blue)
        img[..., 0] -= 20 * blob * alpha
    if lighting:
        g = rng.uniform(0, 2 * np.pi)
        ramp = ((xx - size / 2) * np.cos(g) + (yy - size / 2) * np.sin(g)) / size
        illum = rng.uniform(0.85, 1.1) * (1 + rng.uniform(0.15, 0.35) * ramp) * (1 - 0.25 * (rr / size) ** 2)
        img = img * illum[..., None]
    img = img * exposure + rng.normal(0, noise, img.shape)
    info = dict(defect=defect, centre=(cx, cy), angle_deg=angle, mm_per_px=mm_per_px, outer_d_mm=outer_d,
                defect_box=None if box is None else tuple(float(v) for v in box))
    return np.clip(img, 0, 255).astype(np.uint8), info


def make_dataset(n_per_class=100, classes=CLASSES, seed=0, size=256, **kw):
    """Balanced dataset. Returns (images uint8 [N, H, W, 3], labels int [N], infos list)."""
    rng = np.random.default_rng(seed)
    imgs, labels, infos = [], [], []
    for ci, c in enumerate(classes):
        for _ in range(n_per_class):
            im, inf = make_part(rng.integers(1 << 31), c, size=size, **kw)
            imgs.append(im); labels.append(ci); infos.append(inf)
    order = rng.permutation(len(labels))
    return np.stack(imgs)[order], np.array(labels)[order], [infos[i] for i in order]
