from dataclasses import dataclass, field
from cv2.typing import MatLike, Rect, Point
from cv2 import KalmanFilter
from typing import Optional, List, Dict
from ssbmv.domain.sprite_database import Character
from numpy import ndarray


@dataclass(slots=True)
class Region:
    masked_rgb_slice: MatLike = field(default_factory=list)
    rect: Rect = field(default_factory=lambda: [-1, -1, -1, -1])

    @property
    def centroid(self) -> Point:
        return (self.rect[0] + self.rect[2] // 2, self.rect[1] + self.rect[3] // 2)


@dataclass
class TrackedActor:
    """The persistent state of a fighter tracked across frames."""

    track_id: int  # Unique ID (e.g., Track 0, Track 1)
    player_slot: Optional[int]  # Linked HUD slot (1-4)

    # State & Spatial Tracking
    current_rect: Rect
    predicted_centroid: Optional[Point] = None  # Predicted search window for NEXT frame (from Kalman)
    kalman_filter: KalmanFilter = KalmanFilter()

    # Matching & Classification
    confirmed_character: str = "Unknown"
    confidence_score: float = 0.0

    # Status & Lifecycle
    age_frames: int = 0  # Total frames tracked
    time_since_update: int = 0  # Consecutive missed detections (for deletion check)
    is_active: bool = True  # Active vs Tentative track

    @property
    def centroid(self) -> Point:
        return (
            self.current_rect[0] + self.current_rect[2] // 2,
            self.current_rect[1] + self.current_rect[3] // 2,
        )


@dataclass
class DetectionCandidate:
    """Ephemeral candidate region found by Detector on current frame."""

    rect: Rect
    contour: ndarray  # Raw contour points
    binary_mask: ndarray  # Binary image mask patch
    
    @property
    def centroid(self) -> Point:
        return (
            self.rect[0] + self.rect[2] // 2,
            self.rect[1] + self.rect[3] // 2,
        )

@dataclass
class HUDState:
    player_slot: int  # 1, 2, 3, or 4
    icon_character_id: Optional[str]  # Matched character from HUD icon (e.g., "Fox")
    hud_rect: Rect  # Bounding box of the HUD element at bottom


@dataclass
class GameState:
    """Master frame-level container passed down the pipeline."""

    frame_index: int = 0
    # timestamp_ms: float

    # HUD Metadata (Context Layer)
    hud_states: List[HUDState | None] = field(default_factory=lambda: [None] * 4)

    # Pipeline Processing Layers
    active_tracks: List[TrackedActor] = field(default_factory=list)
    raw_detections: List[DetectionCandidate] = field(default_factory=list)
    debug: bool = False
    huds_found: bool = False

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


# @dataclass(slots=True)
# class TrackedObjectState:
#     track_id: int = -1
#     region: Region
#     rect:rect
#     predicted_centroid: Point = field(default_factory=list)
#     sprite_name: Optional[str]
#     frames_active: int = 0


@dataclass
class Detection:
    character_huds: list[Region]
    character_rois: list[Region]


# @dataclass(slots=True)
# class MatchedCharacter:
#     sprite_name: str = ""


# @dataclass(slots=True)
# class GameState:
#     character_HUDs: list[Region] = field(default=list)
#     characters: list[TrackedObjectState] = field(default_factory=list)
