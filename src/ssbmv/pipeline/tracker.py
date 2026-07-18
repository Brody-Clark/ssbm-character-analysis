from cv2.typing import Rect, Point
from cv2 import KalmanFilter
from src.ssbmv.domain.models import TrackedObjectState, Region
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

    def _match_tracks(self, trackers: list[Point], detections: list[Point]):
        cost_matrix = self._compute_distance_matrix(trackers, detections)

        # Run Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches using a max distance threshold
        MAX_DISTANCE = 75  # TODO: based on ratio: Max pixels a character can realistically travel in 1 frame

        valid_matches = []
        unmatched_trackers = set(range(len(trackers)))
        unmatched_detections = set(range(len(detections)))

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= MAX_DISTANCE:
                valid_matches.append((t, d))
                unmatched_trackers.remove(t)
                unmatched_detections.remove(d)
        
        return valid_matches,unmatched_trackers,unmatched_detections

    def _get_centroid(self, rect: Rect) -> Point:
        x, y, w, h = rect.rect
        return Point((x + w) / 2, (y + h) / 2)

    # TODO: need to capture frames_active and frames_unmatched
    # TODO: tracker shouldnt capture region or sprite name. split detection, track, and match
    def track(
        self, detections: list[Region], prev_tracks: list[TrackedObjectState]
    ) -> list[TrackedObjectState]:
        detection_centroids = [self._get_centroid(d.rect) for d in detections]
        matches, unmatched_tracker_ids, unmatched_detection_ids = self._match_tracks(
            trackers=prev_tracks, detections=detection_centroids
        )

        new_tracks = []
        for prev_track_idx, detection_idx in matches:
            kf = self._kalman_filters[prev_track_idx]
            kf.correct(detection_centroids[detection_idx])
            prev = prev_tracks[prev_track_idx]
            prev.predicted_centroid = kf.predict()
            prev.frames_active += 1
            new_tracks.append(prev)
            self._unmatched_frames[prev_track_idx] = 0  # Reset unmatched frame count

        for id in unmatched_tracker_ids:
            self._unmatched_frames[id] += 1
            if self._unmatched_frames[id] > self._max_unmatched_frames:
                continue
            new_tracks.append(prev_tracks[id])

        for id in unmatched_detection_ids:
            new_tracks.append(
                TrackedObjectState(
                    track_id=0,
                    region=0,
                    predicted_centroid=detection_centroids[id],
                    sprite_name=None,
                    frames_active=1,
                )
            )

        return new_tracks
