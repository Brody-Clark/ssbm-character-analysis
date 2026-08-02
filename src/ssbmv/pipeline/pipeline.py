"""
_summary_
"""

from collections.abc import Callable
from ssbmv.pipeline import detector, tracker, matcher
from ssbmv.domain.models import Frame, TrackedActor, GameState
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import VideoSource
import logging
import json
import cv2 as cv
import time
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, sprite_db: SpriteDatabase, stage: str):
        self._detector: detector.Detector = detector.Detector(stage_name=stage)
        self._tracker: tracker.Tracker = tracker.Tracker()
        self._matcher: matcher.Matcher = matcher.Matcher(sprite_database=sprite_db)

    def _debug_frame(self, frame: Frame, tracked_objs: list[TrackedActor]):
        debug_frame = frame.image.copy()

        for obj in tracked_objs:
            if obj.age_frames < 5:
                continue
            x, y, w, h = obj.current_rect
            cv.rectangle(
                debug_frame, rec=obj.current_rect, color=(220, 220, 32), thickness=2
            )
            cv.circle(
                debug_frame,
                obj.predicted_centroid,
                radius=8,
                color=(20, 220, 200),
                thickness=-1,
            )
            cv.putText(
                debug_frame,
                f"{obj.confirmed_character} - [{obj.confidence_score:.2f}] | predicted: {obj.predicted_centroid[0]:.3f},{obj.predicted_centroid[1]:.3f} ",
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

    def process(self, video_source: VideoSource, output_file_path: Path, debug: bool):
        game_state = GameState()
        game_state.debug = debug
        start = time.perf_counter()
        with open(output_file_path, "w", encoding="utf8") as f:
            while video_source.is_opened():
                game_state.frame_index += 1

                frame: Frame = video_source.read()
                if not frame:
                    break

                detections = self._detector.detect(frame=frame, game_state=game_state)
                game_state.raw_detections = detections

                tracked_actors = self._tracker.track(game_state=game_state)
                game_state.active_tracks = tracked_actors

                tracks = self._matcher.match(frame=frame, game_state=game_state)

                if debug:
                    self._debug_frame(frame=frame, tracked_objs=tracks)

                end = time.perf_counter()
                actors = []
                for tracked_actor in tracked_actors:
                    actors.append(
                        {
                            "current_rect": tracked_actor.current_rect,
                            "predicted_centroid": tracked_actor.predicted_centroid,
                            "confirmed_character": tracked_actor.confirmed_character,
                            "age_frames": tracked_actor.age_frames,
                            "time_since_update": tracked_actor.time_since_update,
                            "is_active": tracked_actor.is_active,
                        }
                    )
                result = {
                    "frame": game_state.frame_index,
                    "expected_player_count": game_state.expected_player_count,
                    "actors": actors,
                    "timestamp": end,
                    "elapsed_frame_time": end - start,
                }
                json.dump(result, f, indent=4)
                # _logger.debug(json.dumps(result))
                start = end

        video_source.release()
        cv.destroyAllWindows()
