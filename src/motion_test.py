import cv2
import os
from datetime import datetime
from camera import OakCamera

SAVE_DIR = "images"
MIN_AREA = 2500

os.makedirs(SAVE_DIR, exist_ok=True)

camera = OakCamera()
camera.start()

prev_gray = None
last_save_time = 0

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

    if motion_detected:
        now = datetime.now().timestamp()

        if now - last_save_time > 5:
            filename = datetime.now().strftime("motion_%Y%m%d_%H%M%S.jpg")
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"Motion detected - saved {path}")
            last_save_time = now

    cv2.imshow("Motion Test", frame)
    prev_gray = gray

    if cv2.waitKey(1) == ord("q"):
        break

camera.stop()