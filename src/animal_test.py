import cv2
import time
from ultralytics import YOLO
from camera import OakCamera

ANIMAL_CLASSES = {"dog", "cat", "horse", "cow", "sheep", "bear", "zebra", "giraffe"}
CONFIDENCE_MIN = 0.60
PRESENT_SECONDS = 2.0
COOLDOWN_SECONDS = 10.0

model = YOLO("yolov8n.pt")
camera = OakCamera()
camera.start()

animal_first_seen = None
last_trigger = 0

while camera.is_running():
    frame = camera.get_frame()
    results = model(frame, verbose=False)[0]

    animal_seen = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in ANIMAL_CLASSES and conf >= CONFIDENCE_MIN:
            animal_seen = True

    annotated = results.plot()

    now = time.time()

    if animal_seen:
        if animal_first_seen is None:
            animal_first_seen = now

        present_time = now - animal_first_seen

        if present_time >= PRESENT_SECONDS and now - last_trigger >= COOLDOWN_SECONDS:
            print("ANIMAL TRIGGER - future sprinkler output")
            last_trigger = now
    else:
        animal_first_seen = None

    cv2.imshow("Animal Trigger Test", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

camera.stop()