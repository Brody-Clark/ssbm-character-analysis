"""End-to-end vision pipeline for actor detection, tracking, and matching."""

import json
import time
from collections import deque, Counter
from dataclasses import asdict
from typing import TextIO
import cv2 as cv
from ssbmv.domain.models import Actor, Frame, GameState
from ssbmv.core.detector import Detector
from ssbmv.core.tracker import Tracker
from ssbmv.core.matcher import Matcher
from ssbmv.source.frame_source import VideoSource

_DEBUG_FRAME_NAME = "SSBMV DEBUG"
_TEMPORAL_CONSISTENCY_FRAMES = 18


class VisionPipeline:
    """Game state prediction pipeline for Super Smash Bros Melee gameplay."""

    def __init__(self, detector: Detector, tracker: Tracker, matcher: Matcher):
        self._detector = detector
        self._tracker = tracker
        self._matcher = matcher

    def _debug_frame(self, frame: Frame, game_state: GameState):
        debug_frame = frame.image.copy()

        for obj in game_state.actors:
            x, y, w, h = obj.rect
            cv.rectangle(debug_frame, rec=obj.rect, color=(220, 220, 32), thickness=2)
            cv.putText(
                debug_frame,
                f"{obj.character_id}",
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

    def _get_best_match(self, matches: list[str]) -> str:
        """Returns most frequent character match or Unknown"""
        if not matches:
            return "Unknown"
        return Counter(matches).most_common(1)[0][0]

    def process(self, video_source: VideoSource, output_stream: TextIO, debug: bool):
        """Runs pipeline for SSBM gameplay source and prints game state results"""
        game_state = GameState()
        game_state.debug = debug
        tracked_matches = {}
        while video_source.is_opened():
            frame: Frame = video_source.read()
            if not frame:
                break
            game_state.frame_index += 1

            start = time.perf_counter()

            detections, huds = self._detector.detect(frame=frame)
            active_tracks, matched_detections = self._tracker.track(detections)
            game_state.hud_states = self._matcher.match_huds(frame=frame, huds=huds)
            matched_actors = self._matcher.match_actors(frame, matched_detections)

            # Temporal consistency check:
            # keep the most frequent character id for a tracked actor
            new_tracked_matches = {}
            for i, track in enumerate(active_tracks):
                prev_matches = tracked_matches.get(track.track_id)
                if prev_matches:
                    cur_match = matched_actors[i]
                    if cur_match is None:
                        prev_matches.append(self._get_best_match(prev_matches))
                    else:
                        prev_matches.append(cur_match.character_id)
                    new_tracked_matches[track.track_id] = prev_matches
                else:
                    new_matches = deque(maxlen=_TEMPORAL_CONSISTENCY_FRAMES)
                    cur_match = matched_actors[i]
                    if cur_match is None:
                        new_matches.append("Unknown")
                    else:
                        new_matches.append(cur_match.character_id)
                    new_tracked_matches[track.track_id] = new_matches
            tracked_matches = new_tracked_matches

            # Set current frame predictions based on results
            game_state.actors.clear()
            for i, track in enumerate(active_tracks):
                uid = track.track_id
                prev_matches = tracked_matches[uid]
                actor_name = self._get_best_match(prev_matches)

                a = Actor()
                a.character_id = actor_name
                matched_actor = matched_actors[i]
                a.animation_id = matched_actor.animation if matched_actor else "Unknown"
                a.rect = active_tracks[i].current_rect
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

        video_source.release()
        if debug:
            cv.destroyWindow(_DEBUG_FRAME_NAME)
