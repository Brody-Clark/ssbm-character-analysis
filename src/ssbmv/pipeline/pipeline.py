"""
_summary_
"""

from collections.abc import Callable
from ssbmv.pipeline import detector, tracker, matcher
from ssbmv.domain.models import Frame, TrackedObjectState, GameState
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import FrameSourceBase
import logging
import json
import cv2 as cv

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

    def _debug_frame(self, frame: Frame, tracked_objs: list[TrackedObjectState]):
        debug_frame = frame.image.copy()

        for obj in tracked_objs:
            x, y, w, h = obj.region.rect
            cv.rectangle(
                debug_frame, rec=obj.region.rect, color=(220, 220, 32), thickness=2
            )
            cv.putText(
                debug_frame,
                f"{obj.sprite_name}",
                org=(x, y - 10),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=2,
            )
        cv.imshow("debug frame", debug_frame)
        while True:
            key = cv.waitKey()
            if key == ord(' '):
                break
        return

    def process(self, frame_source: FrameSourceBase) -> list[TrackedObjectState]:
        predictions = []
        for frame in frame_source:
            frame: Frame = frame_source.get_next_frame()
            
            regions = self._detector.detect(frame=frame, predicted_rois=predictions)

            tracks, track_predictions = self._tracker.track(regions)
            predictions = track_predictions

            tracks = self._matcher.match(frame=frame, tracked_objs=tracks)

            if self._debug_enabled:
                self._debug_frame(frame=frame, tracked_objs=tracks)

            for s in self._subscribers:
                s(tracks)
            game_state = GameState(tracked_objects=tracks, frame_id=frame.frame_id, timestamp=frame.timestamp)
            _logger.info(json.dumps(game_state))
            
        cv.destroyAllWindows()
        return tracks
