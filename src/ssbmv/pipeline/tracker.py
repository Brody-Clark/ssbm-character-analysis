from cv2.typing import Rect, Point
from cv2 import KalmanFilter
from ssbmv.domain.models import (
    TrackedActor,
    GameState,
    DetectionCandidate,
)
import logging
from scipy.optimize import linear_sum_assignment
import numpy as np

_logger = logging.getLogger(__name__)

MAX_DISTANCE = 200  # TODO: based on ratio: Max pixels a character can realistically travel in 1 frame

class Tracker:
    def __init__(self, max_unmatched_frames: int = 4):
        self._max_unmatched_frames = max_unmatched_frames
        self._tracks: list[Point] = []
        self._unmatched_frames: list[int] = 0
        self._kalman_filters: list[KalmanFilter] = []

    def _compute_distance_matrix(self, trackers: list[Point], detections: list[Point]):
        rows, cols = len(trackers), len(detections)
        matrix = np.zeros((rows, cols))
        for i, track in enumerate(trackers):
            for j, det in enumerate(detections):
                matrix[i][j] = np.linalg.norm(np.array(det) - np.array(track))

        return matrix

    def _match_tracks(
        self, trackers: list[TrackedActor], detections: list[DetectionCandidate]
    ):
        track_centroids = [t.centroid for t in trackers]
        detection_centroids = [d.centroid for d in detections]
        cost_matrix = self._compute_distance_matrix(
            track_centroids, detection_centroids
        )

        # Run Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

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
        x, y, w, h = rect
        return ((x + w) / 2, (y + h) / 2)

    def track(self, game_state: GameState) -> list[TrackedActor]:
        matches, unmatched_tracker_ids, unmatched_detection_ids = self._match_tracks(
            trackers=game_state.active_tracks, detections=game_state.raw_detections
        )

        new_tracks = []
        
        # Update matched tracks
        for prev_track_idx, detection_idx in matches:
            track = game_state.active_tracks[int(prev_track_idx)]
            kf = track.kalman_filter
            detection = game_state.raw_detections[int(detection_idx)]
            centroid = np.array(detection.centroid) 
            kf.correct(np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]], dtype=np.float32))

            prediction = kf.predict()
            pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
            track.predicted_centroid = (pred_x, pred_y)
            track.age_frames += 1
            track.time_since_update = 0  # Reset unmatched frame count
            track.current_rect = detection.rect
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
            new_track = TrackedActor(
                                track_id=1,
                                player_slot=None,
                                current_rect=detection.rect,
                                kalman_filter=KalmanFilter(),
                            )
            cur_centroid = self._get_centroid(detection.rect)
            initial_state = np.array([
                [np.float32(cur_centroid[0])],
                [np.float32(cur_centroid[1])],
                [0.0],  # initial vx = 0
                [0.0]   # initial vy = 0
            ], dtype=np.float32)

            # Init new kalman filter
            new_track.kalman_filter.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            new_track.kalman_filter.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            new_track.kalman_filter.statePost = initial_state
            new_track.kalman_filter.statePre = initial_state
            new_track.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
            new_track.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
            new_track.kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)

            prediction = new_track.kalman_filter.predict()
            pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
            new_track.predicted_centroid = (pred_x, pred_y)
            new_tracks.append(new_track)

        return new_tracks
