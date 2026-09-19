import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("money.pt")

# Open the camera (0 = webcam)
cap = cv2.VideoCapture(0)

# Define value for each class
money_values = {
    "rm1": 1,
    "rm20": 20,
    "rm50": 50
}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference
    results = model(frame)

    # Initialize total for this frame
    total_money = 0

    # Extract detections
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]

        # Add value if the class label exists
        if cls_name in money_values:
            total_money += money_values[cls_name]

    # Annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Display the total money on the frame
    cv2.putText(
        annotated_frame,
        f"Total: RM {total_money}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # Show video
    cv2.imshow("YOLO Money Counter", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
