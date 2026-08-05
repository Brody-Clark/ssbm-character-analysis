"""
_summary_
"""

import logging
import json
import time
from typing import TextIO
import cv2 as cv
from ssbmv.pipeline import detector, tracker, matcher
from ssbmv.domain.models import Frame, Track, Actor, DetectionCandidate, HUDDetection, HUDState, GameState
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import VideoSource

_logger = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, sprite_db: SpriteDatabase, stage: str):
        self._detector: detector.Detector = detector.Detector(stage_name=stage)
        self._tracker: tracker.Tracker = tracker.Tracker()
        self._matcher: matcher.Matcher = matcher.Matcher(sprite_database=sprite_db)

    def _debug_frame(self, frame: Frame, tracked_objs: list[Actor]):
        debug_frame = frame.image.copy()

        for obj in tracked_objs:
            if obj.age_frames < 5:
                continue
            x, y, w, h = obj.rect
            cv.rectangle(
                debug_frame, rec=obj.rect, color=(220, 220, 32), thickness=2
            )
            cv.putText(
                debug_frame,
                f"{obj.character_id} - [{obj.confidence_score:.2f}%]",
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

    def process(self, video_source: VideoSource, output_stream: TextIO, debug: bool):
        game_state = GameState()
        game_state.debug = debug
        start = time.perf_counter()
        while video_source.is_opened():
            game_state.frame_index += 1

            frame: Frame = video_source.read()
            if not frame:
                break

            detections, huds = self._detector.detect(frame=frame)
            active_tracks, matched_detections = self._tracker.track(detections)
            matched_actors = self._matcher.match_actors(frame, game_state, active_tracks,matched_detections)
            matched_huds = self._matcher.match_huds(frame=frame, huds)

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
            huds = []
            for hud in game_state.hud_states:
                huds.append(
                    {
                        "player_slot": hud.player_slot,
                        "rect": hud.hud_rect,
                        "character_id": hud.icon_character_id,
                    }
                )
            result = {
                "frame": game_state.frame_index,
                "expected_player_count": game_state.expected_player_count,
                "actors": actors,
                "HUDs": huds,
                "timestamp": end,
                "elapsed_frame_time": end - start,
            }
            json.dump(result, output_stream, indent=4)
            start = end

        video_source.release()
        if debug:
            cv.destroyAllWindows()
