from cv2.typing import Rect
from src.ssbmv.domain.models import TrackedObjectState
import logging

_logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self):
        pass
    
    def track(self, detections: list[Rect]) -> list[TrackedObjectState]:
        pass
    