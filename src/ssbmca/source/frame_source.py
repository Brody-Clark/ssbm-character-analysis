"""Source of video playback"""

import cv2
from ssbmca.domain.models import Frame, Dimension2D


class VideoSource:
    """Source of video data."""

    def __init__(self, video_frame_source: str):
        super().__init__()
        self._capture = cv2.VideoCapture(video_frame_source)
        if not self._capture.isOpened():
            raise RuntimeError("Unable to open video file.")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._dimension = Dimension2D(w=width, h=height)

    def __del__(self):
        self._capture.release()

    def is_opened(self) -> bool:
        """Returns True if file is opened, False otherwise."""
        return self._capture.isOpened()

    def release(self):
        """Releases resources. Should be called once playback is finished."""
        return self._capture.release()

    def read(self) -> Frame | None:
        """Reads next frame and returns a new Frame object or None if reading fails."""
        ret, next_frame = self._capture.read()
        if not ret:
            return None
        ts_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        frame_num = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        frame = Frame(frame_num, next_frame, self._dimension, ts_ms)
        return frame
