import cv2
import depthai as dai


class OakCamera:
    def __init__(self, width=640, height=360):
        self.width = width
        self.height = height
        self.pipeline = None
        self.queue = None

    def start(self):
        self.pipeline = dai.Pipeline()

        cam = self.pipeline.create(dai.node.Camera).build()
        output = cam.requestOutput(
            (self.width, self.height),
            type=dai.ImgFrame.Type.BGR888i
        )

        self.queue = output.createOutputQueue()
        self.pipeline.start()

    def get_frame(self):
        if self.queue is None:
            raise RuntimeError("Camera has not been started.")

        return self.queue.get().getCvFrame()

    def is_running(self):
        return self.pipeline is not None and self.pipeline.isRunning()

    def stop(self):
        cv2.destroyAllWindows()