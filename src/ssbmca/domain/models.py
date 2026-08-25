from dataclasses import dataclass, field
from typing import Optional
from cv2.typing import MatLike, Point, Rect
from numpy import ndarray, float32
import numpy.typing as npt


@dataclass(slots=True)
class Track:
    """Persistent state for one fighter tracked across frames."""

    track_id: int
    current_rect: Rect
    predicted_centroid: Optional[Point] = None
    age_frames: int = 0
    time_since_update: int = 0
    is_active: bool = True
    previous_matches = [None] * 5

    @property
    def centroid(self) -> Point:
        """Return the center of the current bounding rectangle."""
        x, y, w, h = self.current_rect
        return (x + w // 2, y + h // 2)


@dataclass(slots=True)
class Actor:
    """The character prediction produced for a tracked actor in a frame."""

    rect: Rect = field(default_factory=list)
    character_id: str = "Unknown"
    animation_id: str = "Unknown"
    track_id: int = 0


@dataclass(slots=True)
class Match:
    """A candidate character identity with a confidence score."""

    character_id: str = "Unknown"
    animation: str = "Unknown"
    confidence_score: float = 0.0


@dataclass(slots=True)
class DetectionCandidate:
    """An ephemeral actor region found by the detector in the current frame."""

    rect: Rect
    contour: ndarray
    binary_mask: ndarray

    @property
    def centroid(self) -> Point:
        """Return the center of the detection bounding box."""
        x, y, w, h = self.rect
        return (x + w // 2, y + h // 2)


@dataclass(slots=True)
class GameState:
    """Frame-level state passed through the pipeline."""

    frame_index: int = 0
    actors: list[Actor] = field(default_factory=list)
    timestamp_s: float = 0
    elapsed_frame_time_s: float = 0
    debug: bool = False


@dataclass(slots=True)
class Dimension2D:
    """Image dimensions in pixels."""

    w: int
    h: int


@dataclass(slots=True)
class Frame:
    """A single video frame with metadata."""

    frame_id: int
    image: MatLike
    dimensions: Dimension2D
    timestamp: int


@dataclass(slots=True)
class ActorFeatures:
    """Character names, animation states, and corresponding feature vectors."""

    character_names: list[str] = field(default_factory=list)
    animation_names: list[str] = field(default_factory=list)
    features: npt.NDArray[float32] = field(default_factory=list)


@dataclass(slots=True)
class FeatureDatabase:
    """Load and expose sprite assets used by the matching pipeline."""

    actor_features: ActorFeatures
