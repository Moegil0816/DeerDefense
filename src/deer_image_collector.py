import cv2
import os
import time
from datetime import datetime
from camera import OakCamera

EVENT_DIR = "/home/tmagill/DeerDefense/images/deer_collector/events"
HEARTBEAT_DIR = "/home/tmagill/DeerDefense/images/deer_collector/heartbeat"
MIN_AREA = 2500
SAVE_COOLDOWN = 3.0
HEARTBEAT_INTERVAL = 5 * 60

os.makedirs(EVENT_DIR, exist_ok=True)
os.makedirs(HEARTBEAT_DIR, exist_ok=True)


def save_image_atomic(frame, final_path):
    if frame is None or frame.size == 0:
        return False

    tmp_path = f"{final_path}.tmp.jpg"
    try:
        wrote_file = cv2.imwrite(tmp_path, frame)
        if not wrote_file:
            return False

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) <= 0:
            return False

        os.replace(tmp_path, final_path)
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

camera = OakCamera()
camera.start()

prev_gray = None
last_save = 0
last_heartbeat = 0

print("Deer image collector running in headless mode.", flush=True)
print(f"Event directory: {EVENT_DIR}", flush=True)
print(f"Heartbeat directory: {HEARTBEAT_DIR}", flush=True)
print(f"Heartbeat interval: {HEARTBEAT_INTERVAL} seconds", flush=True)

try:
    while camera.is_running():
        frame = camera.get_frame()
        now = time.time()

        if frame is None or frame.size == 0:
            continue

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            heartbeat_name = datetime.now().strftime("heartbeat_%Y%m%d_%H%M%S.jpg")
            heartbeat_path = os.path.join(HEARTBEAT_DIR, heartbeat_name)
            if save_image_atomic(frame, heartbeat_path):
                print(f"Saved heartbeat image: {heartbeat_path}", flush=True)
                last_heartbeat = now

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

        if motion_detected and now - last_save >= SAVE_COOLDOWN:
            event_name = datetime.now().strftime("event_%Y%m%d_%H%M%S.jpg")
            event_path = os.path.join(EVENT_DIR, event_name)
            if save_image_atomic(frame, event_path):
                print(f"Saved event image: {event_path}", flush=True)
                last_save = now

        prev_gray = gray
except KeyboardInterrupt:
    print("Stopping deer image collector.", flush=True)
finally:
    if camera.pipeline is not None and camera.pipeline.isRunning():
        camera.pipeline.stop()