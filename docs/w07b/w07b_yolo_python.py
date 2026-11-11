# MCTA 4364 Machine Vision — Week 7B: YOLO with the Python API
# Full checkpoint file (Steps 1–5). Run with the ▶ Run Python File button in VS Code.
# When an image window opens, click on it and press any key to continue.

# ===== Step 1 — Load a model and predict on one image =====
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")          # downloads the weights the first time
results = model.predict("https://ultralytics.com/images/bus.jpg")

r = results[0]                      # one image -> one Results object
annotated = r.plot()                # numpy image (BGR) with boxes drawn

cv2.imshow("Step 1 - YOLO result", cv2.resize(annotated, None, fx=0.7, fy=0.7))
cv2.waitKey(0)                      # wait until a key is pressed
cv2.destroyAllWindows()

# ===== Step 2 — Look inside the Results object =====
print("Image size (h, w):", r.orig_shape)
print("Number of objects:", len(r.boxes))
print("Class names:", r.names)

for box in r.boxes:
    cls_id = int(box.cls)
    name = r.names[cls_id]
    conf = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{name:10s} conf={conf:.2f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# ===== Step 3 — Filter and count =====
from collections import Counter

# (a) Let YOLO filter for you: only persons (class 0), confidence >= 0.5
results = model.predict("https://ultralytics.com/images/bus.jpg",
                        conf=0.5, classes=[0], verbose=False)
print("Persons found:", len(results[0].boxes))

# (b) Filter yourself, then count every class
results = model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)
r = results[0]
names = [r.names[int(c)] for c, s in zip(r.boxes.cls, r.boxes.conf) if s >= 0.5]
print(Counter(names))

# ===== Step 4 — Draw your own boxes with OpenCV =====
img = r.orig_img.copy()             # original image (BGR), no boxes yet
count = 0

for box in r.boxes:
    if float(box.conf) < 0.5:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    name = r.names[int(box.cls)]
    if name == "person":
        count += 1
    color = (0, 255, 0) if name == "person" else (255, 0, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, f"{name} {float(box.conf):.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.putText(img, f"Persons: {count}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imshow("Step 4 - My own boxes", cv2.resize(img, None, fx=0.7, fy=0.7))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===== Step 5 — Webcam: live person counter (press q to quit) =====
import time

cap = cv2.VideoCapture(0)           # 0 = laptop webcam; or a video path, e.g. "video.mp4"
prev = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(frame, conf=0.5, verbose=False)[0]
    frame = r.plot()

    persons = sum(1 for c in r.boxes.cls if r.names[int(c)] == "person")
    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"Persons: {persons}  FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Step 5 - YOLO webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
