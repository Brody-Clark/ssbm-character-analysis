from ssbmv.domain.models import Frame, Dimension2D
import cv2

class VideoSource():
    def __init__(self, video_frame_source: str):
        super().__init__()
        self._capture = cv2.VideoCapture(video_frame_source)
        if not self._capture.isOpened():
            raise("Unable to open video file.")
        
    def __del__(self):
        self._capture.release()

    def is_opened(self) -> bool:
        return self._capture.isOpened()

    def release(self):
        return self._capture.release()
    
    def read(self) -> Frame | None:
        ret, next_frame =  self._capture.read()
        if not ret:
            return None
        ts_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        frame_num = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame = Frame(frame_num, next_frame, Dimension2D(w=width, h=height), ts_ms)
        return frame