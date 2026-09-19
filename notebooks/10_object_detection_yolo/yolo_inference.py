import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11m.pt")

# Open the video file
video_path = "videos/cars.mp4"
cap = cv2.VideoCapture(video_path)

LINE = [0,400,800,400]

total_car = []
total_bus = []
total_truck = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model.track(
            frame, 
            persist=True,
            tracker = "bytetrack.yaml",
            conf = 0.1,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck (COCO)
            stream=True
            )
        
        for r in results:

            boxes = r.boxes.xyxy.numpy()  # Boxes object for bbox outputs
            ids = r.boxes.id.numpy().astype(int)  # IDs for tracked objects  
            classes = r.boxes.cls.numpy()    # IDs for tracked objects

            for box, id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                # label = f"{model.names[int(cls)]} ID:{int(id)}"
                w,h = x2-x1, y2-y1
                cx,cy = x1 + w//2, y1 + h//2 
                                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"ID {id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if LINE[1] < cy < LINE[1]+35:
                    if id not in (total_car or total_bus or total_truck):
                        if cls == 2:  # Car class
                            total_car.append(id)
                        elif cls == 5:  # Bus class
                            total_bus.append(id)
                        elif cls == 7:  # Truck class
                            total_truck.append(id)

        total_cars = len(total_car)
        total_buses = len(total_bus)
        total_trucks = len(total_truck)

        cv2.line(frame, (LINE[0],LINE[1]), (LINE[2],LINE[3]), (0, 0, 255), 3)
        # Display the annotated frame
        cv2.putText(frame, f"Total cars: {total_cars}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
        cv2.putText(frame, f"Total buses: {total_buses}", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)  
        cv2.putText(frame, f"Total trucks: {total_trucks}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()