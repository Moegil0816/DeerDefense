import cv2
from camera import OakCamera


def main():
    camera = OakCamera()
    camera.start()

    while camera.is_running():
        frame = camera.get_frame()
        cv2.imshow("Deer Defense - OAK-D", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.stop()


if __name__ == "__main__":
    main()