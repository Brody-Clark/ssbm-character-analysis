from cv2.typing import Rect
from src.ssbmv.domain.models import TrackedObjectState, Region
import logging

_logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self):
        pass
    # TODO: remove the list[rect] since trackedObjState has prediction field
    def track(self, detections: list[Region]) -> tuple[list[TrackedObjectState], list[Rect]]:
        pass
    