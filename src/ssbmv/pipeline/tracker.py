import logging
import uuid
import numpy as np
from scipy.optimize import linear_sum_assignment
from cv2.typing import Rect, Point
from cv2 import KalmanFilter
from ssbmv.domain.models import (
    Track,
    DetectionCandidate,
)

_logger = logging.getLogger(__name__)

MAX_DISTANCE = 200
MIN_FRAMES_ACTIVE = 4


class Tracker:
    """Tracks actors and predicts their next centroid location."""

    def __init__(self, max_unmatched_frames: int = 4):
        self._max_unmatched_frames = max_unmatched_frames
        self._tracks: list[Track] = []
        self._unmatched_frames: list[int] = 0
        self._kalman_filters: dict[int, KalmanFilter] = {}

    def _compute_distance_matrix(self, trackers: list[Point], detections: list[Point]):
        rows, cols = len(trackers), len(detections)
        matrix = np.zeros((rows, cols))
        for i, track in enumerate(trackers):
            for j, det in enumerate(detections):
                matrix[i][j] = np.linalg.norm(np.array(det) - np.array(track))

        return matrix

    def _update_tracks(
        self,
        detections: list[DetectionCandidate],
        matches,
        unmatched_tracker_ids,
        unmatched_detection_ids,
    ):
        new_tracks = []
        active_tracks = []
        matched_detections = []
        # Update matched tracks
        for prev_track_idx, detection_idx in matches:
            track = self._tracks[int(prev_track_idx)]
            kf = self._kalman_filters.get(track.track_id, None)
            if kf is None:
                continue
            detection = detections[int(detection_idx)]
            centroid = np.array(detection.centroid)
            kf.correct(
                np.array(
                    [[np.float32(centroid[0])], [np.float32(centroid[1])]],
                    dtype=np.float32,
                )
            )

            prediction = kf.predict()
            pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
            track.predicted_centroid = (pred_x, pred_y)

            track.age_frames += 1
            if track.age_frames > MIN_FRAMES_ACTIVE:
                track.is_active = True
                active_tracks.append(track)
                matched_detections.append(detection)

            track.time_since_update = 0  # Reset unmatched frame count
            track.current_rect = detection.rect
            new_tracks.append(track)

        # Persist unmatched tracks unless they have been unmatched for too long
        for unmatched_id in unmatched_tracker_ids:
            track = self._tracks[unmatched_id]
            track.time_since_update += 1
            if track.time_since_update > self._max_unmatched_frames:
                continue
            new_tracks.append(track)
            if track.is_active:
                active_tracks.append(track)
                matched_detections.append(None)

        # Add new detection tracks
        for unmatched_id in unmatched_detection_ids:
            detection = detections[unmatched_id]
            new_track = Track(
                track_id=uuid.uuid4().int,
                current_rect=detection.rect,
                is_active=False,
            )

            cur_centroid = self._get_centroid(detection.rect)
            initial_state = np.array(
                [
                    [np.float32(cur_centroid[0])],
                    [np.float32(cur_centroid[1])],
                    [0.0],  # initial vx = 0
                    [0.0],  # initial vy = 0
                ],
                dtype=np.float32,
            )

            # Initialize new kalman filter
            kf = KalmanFilter()
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
            kf.measurementMatrix = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
            )
            kf.statePost = initial_state
            kf.statePre = initial_state
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
            kf.errorCovPost = np.eye(4, dtype=np.float32)

            prediction = kf.predict()
            pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
            new_track.predicted_centroid = (pred_x, pred_y)
            new_tracks.append(new_track)
            self._kalman_filters[new_track.track_id] = kf

        self._tracks = new_tracks
        return (active_tracks, matched_detections)

    def _match_tracks(self, detections: list[DetectionCandidate]):
        track_centroids = [t.centroid for t in self._tracks]
        detection_centroids = [d.centroid for d in detections]
        cost_matrix = self._compute_distance_matrix(
            track_centroids, detection_centroids
        )

        # Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        valid_matches = []
        unmatched_trackers = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= MAX_DISTANCE:
                valid_matches.append((t, d))
                unmatched_trackers.remove(t)
                unmatched_detections.remove(d)

        return valid_matches, unmatched_trackers, unmatched_detections

    def _get_centroid(self, rect: Rect) -> Point:
        x, y, w, h = rect
        return ((x + w) / 2, (y + h) / 2)

    def track(
        self, detections: list[DetectionCandidate]
    ) -> tuple[list[Track], list[DetectionCandidate]]:
        """
        Matches detections with nearest active track and creates new tracks for new detections.
        """
        matches, unmatched_tracker_ids, unmatched_detection_ids = self._match_tracks(
            detections=detections
        )
        active_tracks, matched_detections = self._update_tracks(
            detections, matches, unmatched_tracker_ids, unmatched_detection_ids
        )
        return (active_tracks, matched_detections)
