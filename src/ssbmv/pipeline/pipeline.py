"""
_summary_
"""

import logging
import json
import time
from typing import TextIO
import cv2 as cv
from ssbmv.pipeline import detector, tracker, matcher
from ssbmv.domain.models import Frame, Actor, GameState, Dimension2D
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import VideoSource
from dataclasses import asdict

_logger = logging.getLogger(__name__)

MIN_HUD_MATCH_CONFIDENCE = 0.50
MIN_ACTOR_MATCH_CONFIDENCE = 0.50


class VisionPipeline:
    """"""

    def __init__(self, sprite_db: SpriteDatabase, stage: str):
        self._detector: detector.Detector = detector.Detector(stage_name=stage)
        self._tracker: tracker.Tracker = tracker.Tracker()
        self._matcher: matcher.Matcher = matcher.Matcher(sprite_database=sprite_db)

    def _debug_frame(self, frame: Frame, game_state: GameState):
        debug_frame = frame.image.copy()

        for obj in game_state.actors:
            x, y, w, h = obj.rect
            cv.rectangle(debug_frame, rec=obj.rect, color=(220, 220, 32), thickness=2)
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
        """"""
        game_state = GameState()
        game_state.debug = debug
        start = time.perf_counter()
        while video_source.is_opened():
            game_state.frame_index += 1

            frame: Frame = video_source.read()
            if not frame:
                break

            # Pass small copy for faster processing
            # small_img = cv.resize(
            #     frame.image, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR
            # )
            # small_w = int(frame.dimensions.w * 0.5)
            # small_h = int(frame.dimensions.h * 0.5)
            # small_frame = Frame(
            #     frame_id=frame.frame_id,
            #     image=small_img,
            #     dimensions=Dimension2D(w=small_w, h=small_h),
            #     timestamp=frame.timestamp,
            # )

            detections, huds = self._detector.detect(frame=frame)
            # inv_scale = 2.0
            # for det in detections:
            #     det.rect = [int(coord * inv_scale) for coord in det.rect]
            # for hud in huds:
            #     hud.hud_rect = [int(coord * inv_scale) for coord in hud.hud_rect]

            active_tracks, matched_detections = self._tracker.track(detections)
            game_state.hud_states = self._matcher.match_huds(frame=frame, huds=huds)
            matched_actors = self._matcher.match_actors(frame, matched_detections)
            game_state.actors.clear()

            for i, actor in enumerate(matched_actors):
                a = Actor()
                if actor is None:
                    a.character_id = "Unknown"
                    a.rect = active_tracks[i].current_rect
                    a.confidence_score = 0
                else:
                    a.character_id = actor.character_id
                    a.confidence_score = actor.confidence_score
                    a.rect = matched_detections[i].rect
                game_state.actors.append(a)

            if debug:
                self._debug_frame(frame=frame, game_state=game_state)

            end = time.perf_counter()
            game_state.timestamp_s = end
            game_state.elapsed_frame_time_s = end - start

            json.dump(asdict(game_state), output_stream, separators=(",", ":"))
            output_stream.write("\n")

            start = end

        video_source.release()
        if debug:
            cv.destroyAllWindows()
