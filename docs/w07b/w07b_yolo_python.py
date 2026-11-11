# MCTA 4364 Machine Vision — Week 7B: YOLO with the Python API
# Full checkpoint file (Steps 1–4). Run with the ▶ Run Python File button in VS Code.
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

# ===== Step 3 — Draw your own boxes with OpenCV =====
def draw_boxes(img, r, min_conf=0.5):
    """Draw the YOLO boxes on img ourselves. Returns the number of persons."""
    count = 0
    for box in r.boxes:
        conf = float(box.conf)
        if conf < min_conf:                 # skip weak detections
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = r.names[int(box.cls)]
        if name == "person":
            count += 1
        color = (0, 255, 0) if name == "person" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return count

img = r.orig_img.copy()                     # original image (BGR), no boxes yet
persons = draw_boxes(img, r)
cv2.putText(img, f"Persons: {persons}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imshow("Step 3 - My own boxes", cv2.resize(img, None, fx=0.7, fy=0.7))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===== Step 4 — Webcam: live person counter (press q to quit) =====
import time

cap = cv2.VideoCapture(0)                   # 0 = laptop webcam; or a video path, e.g. "video.mp4"
prev = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(frame, verbose=False)[0]
    persons = draw_boxes(frame, r)          # same function as Step 3

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"Persons: {persons}  FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Step 4 - YOLO webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
