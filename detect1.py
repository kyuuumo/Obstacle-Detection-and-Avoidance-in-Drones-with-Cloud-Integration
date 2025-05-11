from ultralytics import YOLO
import mss
import cv2
import numpy as np

model = YOLO(r'C:\object-detection-realtime\best-new.pt')

with mss.mss() as sct:

    monitor = sct.monitors[1]

    save_video = True
    video_filename = "screen_detection_output.avi"
    video_fps = 20
    video_writer = None

    if save_video:
        screen_width = monitor["width"]
        screen_height = monitor["height"]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_filename, fourcc, video_fps, (screen_width, screen_height))

    print("q to stop")

    frame_count = 0

    while True:

        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

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

        save_images = True 
        if save_images:
            cv2.imwrite(f"screen-detects/annotated_frame_{frame_count}.jpg", frame)

        if save_video and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Game Screen Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    if save_video and video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
