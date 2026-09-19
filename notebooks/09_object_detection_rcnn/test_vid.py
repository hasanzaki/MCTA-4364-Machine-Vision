import cv2 
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around each detected face

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "name: Hasan", (x, y-10), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
    
    cv2.putText(frame, f"frame: {len(faces)}", (20, 40), 
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
    
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cv2.waitKey(0)


cap.release()
cv2.destroyAllWindows()
