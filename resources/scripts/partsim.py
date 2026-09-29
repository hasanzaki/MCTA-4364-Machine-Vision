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


# ----------------------------------------------------------------------------------------------------------------
# Multi-object conveyor scenes (detection, data-centric work, tracking)
# ----------------------------------------------------------------------------------------------------------------
SCENE_CLASSES = ["hex_nut", "washer", "bolt", "flange"]
SCENE_MM_PER_PX = 0.5
_TONE = {"hex_nut": 150, "washer": 190, "bolt": 95, "flange": 170}      # zinc, bright steel, black oxide, machined steel


def _sprite_mask(cls, rng, px):
    """Anti-aliasing is handled by the caller. Returns a uint8 mask (supersampled) centred in its own canvas."""
    if cls == "hex_nut":
        af = rng.uniform(16, 18) * px                    # across flats
        n = int(af * 1.3) + 8
        m = np.zeros((n, n), np.uint8)
        r = af / np.sqrt(3)
        pts = [(n / 2 + r * np.cos(a), n / 2 + r * np.sin(a)) for a in np.arange(6) * np.pi / 3]
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
        cv2.circle(m, (n // 2, n // 2), int(5 * px), 0, -1)
    elif cls == "washer":
        od = rng.uniform(22, 26) * px
        n = int(od) + 8
        m = np.zeros((n, n), np.uint8)
        cv2.circle(m, (n // 2, n // 2), int(od / 2), 255, -1)
        cv2.circle(m, (n // 2, n // 2), int(5.3 * px), 0, -1)
    elif cls == "bolt":
        length = rng.uniform(30, 45) * px
        head, shank = 17 * px, 10 * px
        n = int(length + head) + 8
        m = np.zeros((n, n), np.uint8)
        c = n / 2
        top = c - (length + head * 0.8) / 2
        cv2.rectangle(m, (int(c - head / 2), int(top)), (int(c + head / 2), int(top + head * 0.8)), 255, -1)
        cv2.rectangle(m, (int(c - shank / 2), int(top + head * 0.8)), (int(c + shank / 2), int(top + head * 0.8 + length)), 255, -1)
    else:  # flange (the part from make_part, at scene scale)
        R = 20 * px
        n = int(2 * R) + 8
        m = np.zeros((n, n), np.uint8)
        c = n // 2
        cv2.circle(m, (c, c), int(R), 255, -1)
        cv2.rectangle(m, (int(c - 1.5 * px), 0), (int(c + 1.5 * px), int(c - R + 2 * px)), 0, -1)
        cv2.circle(m, (c, c), int(6 * px), 0, -1)
        for k in range(4):
            a = np.pi / 4 + k * np.pi / 2
            cv2.circle(m, (int(c + 14 * px * np.sin(a)), int(c - 14 * px * np.cos(a))), int(2 * px), 0, -1)
    return m


def make_scene(rng=None, n_objects=None, size=(480, 640), classes=SCENE_CLASSES, class_probs=None, min_visible=0.5,
               lighting=True, noise=3.0, exposure=1.0, return_masks=False):
    """A conveyor scene with several parts. Returns (bgr image, boxes [N, 4] xyxy float, labels [N] int[, masks]).

    Objects may overlap; later objects occlude earlier ones. Boxes are the bounding boxes of the *visible* pixels and
    objects less than `min_visible` visible are dropped (as an annotator would).
    """
    rng = np.random.default_rng(rng)
    H, W = size
    n_objects = int(rng.integers(3, 9)) if n_objects is None else n_objects
    px = _SS / SCENE_MM_PER_PX
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    belt = 50 + _brushed_texture(rng, H, W, 0, 5) + rng.normal(0, 2, (H, W))
    img = belt.copy()
    shade = np.ones((H, W), np.float32)
    owner = np.full((H, W), -1, np.int32)
    labels, full_area = [], []
    for k in range(n_objects):
        ci = int(rng.choice(len(classes), p=class_probs))
        cls = classes[ci]
        m = _sprite_mask(cls, rng, px)
        rot = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), rng.uniform(0, 360), 1.0)
        m = cv2.warpAffine(m, rot, m.shape[::-1])
        m = cv2.resize(m, (m.shape[1] // _SS, m.shape[0] // _SS), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        h, w = m.shape
        x0, y0 = int(rng.integers(-w // 4, W - 3 * w // 4)), int(rng.integers(-h // 4, H - 3 * h // 4))
        full = np.zeros((H, W), np.float32)
        ys, xs = slice(max(y0, 0), min(y0 + h, H)), slice(max(x0, 0), min(x0 + w, W))
        full[ys, xs] = m[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
        if full.sum() < 20:
            continue
        # soft shadow on whatever is below
        sh = cv2.GaussianBlur(np.roll(full, (4, 3), (0, 1)), (0, 0), 3)
        shade *= 1 - 0.45 * sh
        tex = _TONE[cls] + _brushed_texture(rng, H, W, rng.uniform(0, 180), 10)
        if cls in ("washer", "flange"):
            tex = tex + 10 * np.cos(np.hypot(xx - (x0 + w / 2), yy - (y0 + h / 2)) / 3.0) * 0.3
        img = img * shade * (1 - full) + tex * full
        shade = np.where(full > 0.5, 1.0, shade)
        owner[full > 0.5] = len(labels)
        labels.append(ci)
        full_area.append((full > 0.5).sum())
    boxes, keep, masks = [], [], []
    for i in range(len(labels)):
        vis = owner == i
        if vis.sum() < min_visible * full_area[i] or vis.sum() < 15:
            continue
        ys_, xs_ = np.nonzero(vis)
        boxes.append([xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1]); keep.append(i); masks.append(vis)
    img = np.stack([img * 0.97, img, img * 1.03], -1)
    if lighting:
        g = rng.uniform(0, 2 * np.pi)
        ramp = ((xx - W / 2) * np.cos(g) + (yy - H / 2) * np.sin(g)) / max(H, W)
        img = img * (rng.uniform(0.9, 1.08) * (1 + 0.25 * ramp))[..., None]
    img = np.clip(img * exposure + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    out = (img, np.array(boxes, np.float32).reshape(-1, 4), np.array([labels[i] for i in keep], np.int64))
    return out + (np.array(masks, bool).reshape(-1, H, W),) if return_masks else out


def draw_boxes(img, boxes, labels, scores=None, class_names=SCENE_CLASSES, colors=None, thickness=2):
    """Draw xyxy boxes with class names (and scores) on a copy of a BGR image."""
    colors = colors or [(66, 133, 244), (52, 168, 83), (234, 67, 53), (251, 188, 5), (171, 71, 188), (0, 172, 193)]
    vis = img.copy()
    for i, (b, l) in enumerate(zip(boxes, labels)):
        c = colors[int(l) % len(colors)]
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, thickness)
        txt = class_names[int(l)] + ("" if scores is None else f" {scores[i]:.2f}")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x0, max(y0 - th - 4, 0)), (x0 + tw + 2, max(y0, th + 4)), c, -1)
        cv2.putText(vis, txt, (x0 + 1, max(y0 - 3, th + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
