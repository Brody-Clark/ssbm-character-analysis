from cv2.typing import Rect, Point
from cv2 import KalmanFilter
from src.ssbmv.domain.models import (
    Region,
    TrackedActor,
    GameState,
    DetectionCandidate,
)
import logging
from scipy.optimize import linear_sum_assignment
import math
import numpy as np

_logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, max_unmatched_frames: int = 4):
        self._max_unmatched_frames = max_unmatched_frames
        self._tracks: list[Point] = []
        self._unmatched_frames: list[int] = 0
        self._kalman_filters: list[KalmanFilter] = []

    def _compute_distance_matrix(self, trackers: list[Point], detections: list[Point]):
        rows, cols = len(trackers), len(detections)
        matrix = [[0 for _ in range(cols)] for _ in range(rows)]
        for i, track in enumerate(trackers):
            for j, det in enumerate(detections):
                matrix[i][j] = np.linalg.norm(det - track)

        return matrix

    def _match_tracks(
        self, trackers: list[TrackedActor], detections: list[DetectionCandidate]
    ):
        track_centroids = [t.centroid() for t in trackers]
        detection_centroids = [d.centroid() for d in detections]
        cost_matrix = self._compute_distance_matrix(
            track_centroids, detection_centroids
        )

        # Run Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches using a max distance threshold
        MAX_DISTANCE = 120  # TODO: based on ratio: Max pixels a character can realistically travel in 1 frame

        valid_matches = []
        unmatched_trackers = set(range(len(trackers)))
        unmatched_detections = set(range(len(detections)))

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= MAX_DISTANCE:
                valid_matches.append((t, d))
                unmatched_trackers.remove(t)
                unmatched_detections.remove(d)

        return valid_matches, unmatched_trackers, unmatched_detections

    def _get_centroid(self, rect: Rect) -> Point:
        x, y, w, h = rect.rect
        return Point((x + w) / 2, (y + h) / 2)

    def track(self, game_state: GameState) -> list[TrackedActor]:
        matches, unmatched_tracker_ids, unmatched_detection_ids = self._match_tracks(
            trackers=game_state.active_tracks, detections=game_state.raw_detections
        )

        new_tracks = []
        
        # Update matched tracks
        for prev_track_idx, detection_idx in matches:
            track = game_state.active_tracks[prev_track_idx]
            kf = track.kalman_filter
            kf.correct(game_state.raw_detections[detection_idx].centroid())
            track.predicted_centroid = kf.predict()
            track.age_frames += 1
            track.time_since_update = 0  # Reset unmatched frame count
            new_tracks.append(track)

        # Persist unmatched tracks unless they have been unmatched for too long
        for id in unmatched_tracker_ids:
            track = game_state.active_tracks[id]
            track.time_since_update += 1
            if track.time_since_update > self._max_unmatched_frames:
                continue
            new_tracks.append(track)

        # Add new detection tracks
        for id in unmatched_detection_ids:
            detection = game_state.raw_detections[id]
            new_tracks.append(
                TrackedActor(
                    track_id=1,
                    player_slot=None,
                    current_rect=detection.rect,
                    kalman_filter=KalmanFilter(),
                )
            )

        return new_tracks
