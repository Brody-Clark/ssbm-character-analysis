from dataclasses import dataclass, field
from cv2.typing import MatLike, Rect
from typing import Optional

@dataclass(slots=True)
class Region:
    masked_rgb_slice: MatLike = field(default_factory=list)
    rect: Rect = [-1,-1,-1,-1]

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
    region: Region
    predicted_region: Rect
    sprite_name: Optional[str]
