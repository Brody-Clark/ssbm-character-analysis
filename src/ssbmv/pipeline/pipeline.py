"""
_summary_
"""

from collections.abc import Callable
from ssbmv.pipeline import detector, tracker, matcher
from ssbmv.domain.models import Frame, TrackedObjectState
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import FrameSourceBase
import logging

_logger = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, sprite_db: SpriteDatabase, debug_enabled: bool = False):
        self._detector: detector.Detector = detector.Detector()
        self._tracker: tracker.Tracker = tracker.Tracker()
        self._matcher: matcher.Matcher = matcher.Matcher(sprite_database=sprite_db)
        self._subscribers = []
        self._debug_enabled = debug_enabled

    def subscribe(self, callback: Callable[..., None]) -> None:
        if not callable(callback):
            raise TypeError(
                f"Expected a callable, but received {type(callback).__name__}"
            )

        self._subscribers.append(callback)

    def debug_frame(self, frame: Frame, tracked_objs: list[TrackedObjectState]):
        pass

    def process(self, frame_source: FrameSourceBase) -> list[TrackedObjectState]:
        predictions = []
        for frame in frame_source:
            frame = frame_source.get_next_frame()

            regions = self._detector.detect(frame=frame, predicted_rois=predictions)

            tracks, track_predictions = self._tracker.track(regions)
            predictions = track_predictions

            tracks = self._matcher.match(tracked_objs=tracks)

            if self._debug_enabled:
                self.debug_frame(frame=frame, tracked_objs=tracks)

            for s in self._subscribers:
                s(tracks)

        return tracks
