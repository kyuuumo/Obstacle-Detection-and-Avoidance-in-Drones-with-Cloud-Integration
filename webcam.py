from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO(r'C:\object-detection-realtime\best-new.pt')

output_dir = r'C:\object-detection-realtime\screen-detects'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error")
    exit()

print("q to stop")

frame_count = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    resized_frame = cv2.resize(frame, (640, 640))

    results = model(resized_frame, conf=0.7)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()
        names = results[0].names

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            label = f"{names[int(cls)]} {confidences[classes.index(cls)]:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_filename, frame)

    print(f"Saved frame: {output_filename}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
