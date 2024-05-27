from ultralytics import YOLO
from PIL import Image
import cv2
import torch

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    print("Found GPU!")
    torch.cuda.set_device(0)


model = YOLO("yolov8n.pt")

img = cv2.imread("tabletop/1.jpg")


def get_ball_positions(img):
    imgsz = 2560
    results = model.predict(source=img, imgsz=imgsz, conf=0.01)

    height = img.shape[0]
    width = img.shape[1]
    length = max((height, width))
    scale = length / imgsz

    def proc_box(box):
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        return (x1, y1, x2, y2)

    def filter_box(box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        max_allowed_size = (y1 / height) * 30 + 40

        if w > max_allowed_size or h > max_allowed_size:
            return False

        if w < 10 or h < 10:
            return False

        ratio = w / h
        if ratio < 0.5 or ratio > 2:
            return False

        return True

    def box_contains(contained, container):
        if contained == container:
            return False

        if (
            container[0] < contained[0]
            and container[1] < contained[1]
            and container[2] > contained[2]
            and container[3] > contained[3]
        ):
            return True

        mx1 = max(container[0], contained[0])
        my1 = max(container[1], contained[1])
        mx2 = min(container[2], contained[2])
        my2 = min(container[3], contained[3])

        dx = max(0, mx2 - mx1)
        dy = max(0, my2 - my1)

        area_shared = dx * dy
        area_self = (contained[3] - contained[1]) * (contained[2] - contained[0])
        area_other = (container[3] - container[1]) * (container[2] - container[0])

        if area_shared / area_self > 0.9:
            return area_other > area_self

        return False

    ball_circles = []

    for result in results:
        boxes = result.boxes

        boxes = list(map(proc_box, boxes))
        boxes = list(filter(filter_box, boxes))

        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            if any(box_contains(box, b) for b in boxes):
                color = (255, 0, 0)
                continue
            else:
                color = (0, 255, 0)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = max(x2 - x1, y2 - y1) // 2
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # cv2.circle(img, center, radius, color, 2)
            ball_circles.append((center, radius))

    return ball_circles


cv2.imshow("image", img)
cv2.waitKey(0)
