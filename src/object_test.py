import cv2
from ultralytics import YOLO
from camera import OakCamera

model = YOLO("yolov8n.pt")

camera = OakCamera()
camera.start()

while camera.is_running():
    frame = camera.get_frame()

    results = model(frame, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("Object Detection Test", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

camera.stop()