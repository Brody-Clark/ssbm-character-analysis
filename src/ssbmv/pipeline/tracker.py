from cv2.typing import Rect
from src.ssbmv.domain.models import TrackedObjectState, Region
import logging
from scipy.optimize import linear_sum_assignment
import math

_logger = logging.getLogger(__name__)

class Tracker:
    def __init__(self):
        self._tracks = []
        self._kalman_filters = []
        
    def _compute_distance_matrix(self, trackers, detections):
        
        def _get_euclidean_dist(c1, c2):
            return math.sqrt(c1*c1 + c2*c2)
        
        rows, cols = len(trackers), len(detections)
        matrix = [[0 for _ in range(cols)] for _ in range(rows)]
        for i,t in enumerate(trackers):
            for j,d in enumerate(detections):
                matrix[i][j] = _get_euclidean_dist(t, d) # TODO: need centroid not rect here
        
        return matrix
            
                
    def _match_tracks(self, trackers, detections):
        cost_matrix = self._compute_distance_matrix(trackers, detections)

        # 2. Run Hungarian Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Filter matches using a physical distance threshold
        MAX_DISTANCE = 75  # TODO: based on ratio: Max pixels a character can realistically travel in 1 frame

        valid_matches = []
        unmatched_trackers = set(range(N))
        unmatched_detections = set(range(M))

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= MAX_DISTANCE:
                valid_matches.append((t, d))
                unmatched_trackers.remove(t)
                unmatched_detections.remove(d)
                
                
    # TODO: remove the list[rect] since trackedObjState has prediction field
    def track(self, detections: list[Region]) -> tuple[list[TrackedObjectState], list[Rect]]:
        pass
    