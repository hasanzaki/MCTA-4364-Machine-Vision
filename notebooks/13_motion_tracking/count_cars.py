import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11m.pt")

# Video
vid = cv2.VideoCapture("videos/cars.mp4")

# Mask
# mask = cv2.imread("videos/mask.png")

# Counting line
LINE = [300, 300, 750, 300]

totalCount = []

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # imgRegion = cv2.bitwise_and(frame, mask)

    # 🔥 YOLO tracking (ByteTrack by default)
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5,
        classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck (COCO)
        stream=True
    )

    for r in results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.numpy()
        ids = r.boxes.id.numpy().astype(int)
        clss = r.boxes.cls.numpy().astype(int)

        for box, track_id, cls_idx in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
                2
            )

            # 🚗 Counting logic
            if LINE[0] < cx < LINE[2] and LINE[1] < cy < LINE[1] + 20:
                if track_id not in totalCount:
                    totalCount.append(track_id)

    # Draw counting line
    cv2.line(frame, (LINE[0], LINE[1]), (LINE[2], LINE[3]), (0, 0, 255), 4)

    cv2.putText(
        frame,
        f"Number of vehicles: {len(totalCount)}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

vid.release()
cv2.destroyAllWindows()
