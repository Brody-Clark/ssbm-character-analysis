"""End-to-end vision pipeline for actor detection, tracking, and matching."""

import json
import time
from dataclasses import asdict
from typing import TextIO
import cv2 as cv
from ssbmv.domain.models import Actor, Frame, GameState
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.core import detector, matcher, tracker
from ssbmv.source.frame_source import VideoSource

_DEBUG_FRAME_NAME = "SSBMV DEBUG"


class VisionPipeline:
    """Game state prediction pipeline for Super Smash Bros Melee gameplay."""

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
        for hud in game_state.hud_states:
            if hud is None:
                continue
            x, y, w, h = hud.hud_rect
            cv.rectangle(
                debug_frame, rec=[x, y, w, h], color=(158, 200, 22), thickness=2
            )
            cv.putText(
                debug_frame,
                f"{hud.icon_character_id}",
                org=(x, y - 10),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0),
                thickness=2,
            )
        cv.imshow(_DEBUG_FRAME_NAME, debug_frame)
        while True:
            key = cv.waitKey()
            if key == ord(" "):
                break
        return

    def process(self, video_source: VideoSource, output_stream: TextIO, debug: bool):
        """Runs pipeline for SSBM gameplay source and prints game state results"""
        game_state = GameState()
        game_state.debug = debug

        while video_source.is_opened():
            frame: Frame = video_source.read()
            if not frame:
                break
            game_state.frame_index += 1
            
            start = time.perf_counter()
            # Run detection -> Tracking -> Matching
            detections, huds = self._detector.detect(frame=frame)
            active_tracks, matched_detections = self._tracker.track(detections)
            game_state.hud_states = self._matcher.match_huds(frame=frame, huds=huds)
            matched_actors = self._matcher.match_actors(frame, matched_detections)

            # Set current frame predictions based on results
            game_state.actors.clear()
            for i, actor in enumerate(matched_actors):
                a = Actor()
                if actor is None:
                    a.character_id = "Unknown"
                    a.rect = active_tracks[i].current_rect
                    a.confidence_score = 0
                    a.track_id = active_tracks[i].track_id
                else:
                    a.character_id = actor.character_id
                    a.confidence_score = actor.confidence_score
                    a.rect = matched_detections[i].rect
                    a.track_id = active_tracks[i].track_id
                game_state.actors.append(a)

            end = time.perf_counter()

            game_state.timestamp_s = end
            game_state.elapsed_frame_time_s = end - start

            if debug:
                self._debug_frame(frame=frame, game_state=game_state)

            json.dump(asdict(game_state), output_stream, separators=(",", ":"))
            output_stream.write("\n")

            start = end

        # Cleanup
        video_source.release()
        if debug:
            cv.destroyWindow(_DEBUG_FRAME_NAME)
