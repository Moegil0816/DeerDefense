import cv2
import os
import time
from datetime import datetime
from camera import OakCamera

SAVE_DIR = "images/deer_collector"
MIN_AREA = 2500
SAVE_COOLDOWN = 3.0

os.makedirs(SAVE_DIR, exist_ok=True)

camera = OakCamera()
camera.start()

prev_gray = None
last_save = 0

print("Deer image collector running. Press q to quit.")

while camera.is_running():
    frame = camera.get_frame()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray is None:
        prev_gray = gray
        continue

    diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    now = time.time()

    if motion_detected and now - last_save >= SAVE_COOLDOWN:
        filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        print(f"Saved {path}")
        last_save = now

    cv2.imshow("Deer Image Collector", frame)

    prev_gray = gray

    if cv2.waitKey(1) == ord("q"):
        break

camera.stop()