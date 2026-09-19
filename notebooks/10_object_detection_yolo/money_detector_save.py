import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("money.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Money mapping
money_values = {
    "rm1": 1,
    "rm20": 20,
    "rm50": 50
}

# Frame properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30

# Use XVID for Windows compatibility
out = cv2.VideoWriter(
    "output_money.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height)
)

start_time = time.time()
duration = 20

while cap.isOpened():

    if time.time() - start_time > duration:
        break

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    total_money = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        if cls_name in money_values:
            total_money += money_values[cls_name]

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        f"Total: RM {total_money}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    out.write(annotated)

    # display frame (only works in native Python, not in Jupyter)
    cv2.imshow("YOLO Money Counter", annotated)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
