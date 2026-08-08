from dataclasses import dataclass, field
from typing import Optional, List
from cv2.typing import MatLike, Rect, Point
from numpy import ndarray


@dataclass
class Track:
    """The persistent state of a fighter tracked across frames."""

    track_id: int
    current_rect: Rect
    predicted_centroid: Optional[Point] = None
    age_frames: int = 0
    time_since_update: int = 0
    is_active: bool = True

    @property
    def centroid(self) -> Point:
        """Returns centroid (x,y) of current bounding rect."""
        return (
            self.current_rect[0] + self.current_rect[2] // 2,
            self.current_rect[1] + self.current_rect[3] // 2,
        )


@dataclass
class Actor:
    """Character being tracked and identified in a frame"""

    rect: Rect = field(default_factory=list)
    character_id: str = "Unknown"
    confidence_score: float = 0.0
    track_id: int = 0


@dataclass
class Match:
    character_id: str = "Unknown"
    confidence_score: float = 0.0


@dataclass
class DetectionCandidate:
    """Ephemeral candidate region found by Detector on current frame."""

    rect: Rect
    contour: ndarray
    binary_mask: ndarray

    @property
    def centroid(self) -> Point:
        return (
            self.rect[0] + self.rect[2] // 2,
            self.rect[1] + self.rect[3] // 2,
        )


@dataclass
class HUDDetection:
    """"""

    player_slot: int
    hud_rect: Rect
    binary_mask: MatLike


@dataclass
class HUDState:
    """"""

    player_slot: int
    icon_character_id: Optional[str]
    hud_rect: Rect


@dataclass
class GameState:
    """Master frame-level container passed down the pipeline."""

    frame_index: int = 0
    hud_states: List[HUDState] = field(default_factory=list)
    actors: List[Actor] = field(default_factory=list)
    timestamp_s: float = 0
    elapsed_frame_time_s: float = 0
    debug: bool = False

    @property
    def expected_player_count(self) -> int:
        """Helper to tell pipeline how many active fighters should exist."""
        return sum([1 for h in self.hud_states if h is not None])


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
