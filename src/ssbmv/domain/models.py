from dataclasses import dataclass
from cv2.typing import MatLike, Rect
from typing import Optional


@dataclass(slots=True)
class Dimension2D:
    w: int
    h: int

@dataclass(slots=True)
class Frame:
    frame_id: int
    image: MatLike
    dimensions: Dimension2D
    timestamp: int

@dataclass(slots=True)
class TrackedObjectState:
    track_id: int = -1
    rect: Rect = [-1,-1,-1,-1]
    object_name: Optional[str]
    animation_name: Optional[str]