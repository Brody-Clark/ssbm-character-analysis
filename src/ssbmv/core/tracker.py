"""Track detections across frames with a simple Kalman filter-based tracker."""

import uuid
import numpy as np
from cv2 import KalmanFilter
from cv2.typing import Point, Rect
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ssbmv.domain.models import DetectionCandidate, Track

MAX_DISTANCE = 200
MIN_FRAMES_ACTIVE = 4


class Tracker:
    """Track actors and predict their next centroid location."""

    def __init__(self, max_unmatched_frames: int = 4):
        self._max_unmatched_frames = max_unmatched_frames
        self._tracks: list[Track] = []
        self._kalman_filters: dict[int, KalmanFilter] = {}

    def _compute_distance_matrix(self, trackers: list[Point], detections: list[Point]):
        """Return the pairwise Euclidean distance matrix between tracks and detections."""
        if not trackers or not detections:
            return np.zeros((len(trackers), len(detections)))
        return cdist(trackers, detections, metric="euclidean")

    def _build_kalman_filter(self, centroid: Point) -> KalmanFilter:
        """Create and initialize a Kalman filter at the given centroid."""
        initial_state = np.array(
            [
                [np.float32(centroid[0])],
                [np.float32(centroid[1])],
                [0.0],
                [0.0],
            ],
            dtype=np.float32,
        )

        kf = KalmanFilter()
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.statePost = initial_state
        kf.statePre = initial_state
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        _ = kf.predict() # Needed to establish initial state
        return kf

    def _update_tracks(
        self,
        detections: list[DetectionCandidate],
        matches,
        unmatched_tracker_ids,
        unmatched_detection_ids,
    ):
        """Refresh existing tracks, keep unmatched ones alive until limit is met, and add new ones."""
        new_tracks, active_tracks, matched_detections = [], [], []

        for prev_track_idx, detection_idx in matches:
            track = self._tracks[int(prev_track_idx)]
            kf = self._kalman_filters.get(track.track_id)
            if kf is None:
                continue

            detection = detections[int(detection_idx)]
            measurement = np.array(
                [
                    [np.float32(detection.centroid[0])],
                    [np.float32(detection.centroid[1])],
                ],
                dtype=np.float32,
            )
            kf.correct(measurement)

            prediction = kf.predict()
            track.predicted_centroid = (
                int(prediction[0][0]),
                int(prediction[1][0]),
            )
            track.age_frames += 1
            track.time_since_update = 0
            track.current_rect = detection.rect

            if track.age_frames > MIN_FRAMES_ACTIVE:
                track.is_active = True
                active_tracks.append(track)
                matched_detections.append(detection)

            new_tracks.append(track)

        for unmatched_id in unmatched_tracker_ids:
            track = self._tracks[unmatched_id]
            track.time_since_update += 1
            if track.time_since_update > self._max_unmatched_frames:
                continue

            new_tracks.append(track)
            if track.is_active:
                active_tracks.append(track)
                matched_detections.append(None)

        for unmatched_id in unmatched_detection_ids:
            detection = detections[unmatched_id]
            new_track = Track(
                track_id=uuid.uuid4().int,
                current_rect=detection.rect,
                is_active=False,
            )
            new_track.predicted_centroid = self._get_centroid(
                detection.rect
            )
            new_tracks.append(new_track)
            self._kalman_filters[new_track.track_id] = self._build_kalman_filter(
                self._get_centroid(detection.rect)
            )

        active_track_ids = {track.track_id for track in new_tracks}
        self._kalman_filters = {
            tid: kf
            for tid, kf in self._kalman_filters.items()
            if tid in active_track_ids
        }

        self._tracks = new_tracks
        return active_tracks, matched_detections
    
    def _match_tracks(self, detections: list[DetectionCandidate]):
        """Match the current detection set to existing tracks using Hungarian assignment."""
        track_centroids = [track.centroid for track in self._tracks]
        detection_centroids = [detection.centroid for detection in detections]
        cost_matrix = self._compute_distance_matrix(
            track_centroids, detection_centroids
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        valid_matches = []
        unmatched_trackers = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))

        for track_idx, detection_idx in zip(row_ind, col_ind):
            if cost_matrix[track_idx, detection_idx] <= MAX_DISTANCE:
                valid_matches.append((track_idx, detection_idx))
                unmatched_trackers.remove(track_idx)
                unmatched_detections.remove(detection_idx)

        return valid_matches, unmatched_trackers, unmatched_detections

    def _get_centroid(self, rect: Rect) -> Point:
        """Return the center of a rectangle."""
        x, y, w, h = rect
        return (x + w / 2, y + h / 2)

    def track(
        self, detections: list[DetectionCandidate]
    ) -> tuple[list[Track], list[DetectionCandidate | None]]:
        """Match detections to tracks and create new tracks when needed."""
        matches, unmatched_tracker_ids, unmatched_detection_ids = self._match_tracks(
            detections=detections
        )
        return self._update_tracks(
            detections,
            matches,
            unmatched_tracker_ids,
            unmatched_detection_ids,
        )
