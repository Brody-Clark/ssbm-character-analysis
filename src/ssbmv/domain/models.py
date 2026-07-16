from dataclasses import dataclass, field
from cv2.typing import MatLike, Rect, Point
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
    predicted_centroid: Point = field(default_factory=list)
    sprite_name: Optional[str]
    frames_active: int = 0
    
@dataclass(slots=True)
class GameState:
    tracked_objects: list[TrackedObjectState] = field(default_factory=list)
    frame_id: int = 0
    timestamp: int = 0
