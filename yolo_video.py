from ultralytics import YOLO
import cv2
import torch

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    print("Found GPU!")
    torch.cuda.set_device(0)

# Initialize the model and set the image size
model = YOLO("yolov8n.pt")
imgsz = 640

# Setup video read and write
input_video = cv2.VideoCapture("data/vid1.mp4")  # Set the path to your video file
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    "output.avi", fourcc, 20.0, (int(input_video.get(3)), int(input_video.get(4)))
)

import time

# Process each frame
while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    t0 = time.time()
    results = model.predict(source=frame, imgsz=imgsz, conf=0.01, verbose=False)
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.4f} seconds")

    height, width = frame.shape[:2]
    length = max(height, width)
    scale = length / imgsz

    for result in results:
        boxes = result.boxes

        print(len(boxes))

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = (
                int(x1 * scale),
                int(y1 * scale),
                int(x2 * scale),
                int(y2 * scale),
            )
            w, h = x2 - x1, y2 - y1

            max_allowed_size = (y1 / height) * 30 + 40

            if w > max_allowed_size or h > max_allowed_size or w < 10 or h < 10:
                continue

            ratio = w / h
            if ratio < 0.5 or ratio > 2:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

# Release resources
input_video.release()
out.release()
cv2.destroyAllWindows()
