from ultralytics import YOLO
import cv2

# Load lightweight YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Lower resolution for smooth performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Human & Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#Press Q to exit.