"""
_summary_
"""

from collections.abc import Callable
from ssbmv.pipeline import detector, tracker, matcher
from ssbmv.domain.models import Frame, TrackedActor, GameState
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import FrameSourceBase
import logging
import json
import cv2 as cv
import time

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

    def _debug_frame(self, frame: Frame, tracked_objs: list[TrackedActor]):
        debug_frame = frame.image.copy()

        for obj in tracked_objs:
            x, y, w, h = obj.current_rect
            cv.rectangle(
                debug_frame, rec=obj.current_rect, color=(220, 220, 32), thickness=2
            )
            cv.rectangle(
                debug_frame, rec=obj.predicted_centroid, color=(20, 220, 200), thickness=1
            )
            cv.putText(
                debug_frame,
                f"{obj.confirmed_character}",
                org=(x, y - 10),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=2,
            )
        cv.imshow("debug frame", debug_frame)
        while True:
            key = cv.waitKey()
            if key == ord(" "):
                break
        return


    def process(self, frame_source: FrameSourceBase):
        game_state = GameState()
        start = time.perf_counter()
        for frame in frame_source:
            game_state.frame_index += 1

            frame: Frame = frame_source.get_next_frame()

            detections = self._detector.detect(frame=frame, game_state=game_state)
            game_state.raw_detections = detections

            tracked_actors = self._tracker.track(game_state=game_state)
            game_state.active_tracks = tracked_actors

            tracks = self._matcher.match(frame=frame, game_state=game_state)

            if self._debug_enabled:
                self._debug_frame(frame=frame, tracked_objs=tracks)


            end = time.perf_counter()
            result = {
                "frame": game_state.frame_index,
                "expected_player_count": game_state.expected_player_count,
                "active_tracks": game_state.active_tracks,
                "timestamp": end,
                "elapsed_frame_time": end - start
            }
            _logger.info(json.dumps(result))
            start = end

        cv.destroyAllWindows()
