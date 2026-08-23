"""Feature extraction and template matching for actors and HUD icons."""

import logging
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from ssbmv.domain.models import (
    DetectionCandidate,
    Frame,
    Match,
    FeatureDatabase,
)
from ssbmv.core.feature_extractor import FeatureExtractor

_logger = logging.getLogger(__name__)

MATCH_CONFIDENCE_THRESHOLD = 0.65


class Matcher:
    """Build feature templates and match detections against them."""

    def __init__(
        self, feature_extractor: FeatureExtractor, feature_database: FeatureDatabase
    ):
        self._extractor = feature_extractor
        self._feature_db = feature_database

    def _distance_to_confidence(self, dist: float) -> float:
        """Convert a cosine distance into a confidence score."""
        return max(0.0, 1.0 - dist)

    def match_actors(
        self, frame: Frame, tracked_detections: list[DetectionCandidate]
    ) -> list[Match | None]:
        """Match tracked actor detections against the precompiled character templates."""
        matches = [None] * len(tracked_detections)
        matched_detection_indices = []
        query_features = []

        for idx, detection in enumerate(tracked_detections):
            if detection is None:
                continue

            x, y, w, h = detection.rect
            cropped_image = frame.image[y : y + h, x : x + w]
            if cropped_image.size == 0:
                continue
            # Get masked RGB image to match with template
            masked_img = cv.bitwise_and(
                cropped_image, cropped_image, mask=detection.binary_mask
            )
            success, features = self._extractor.get_character_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_detection_indices.append(idx)

        if not query_features:
            return matches

        query_matrix = np.array(query_features, dtype=np.float32)
        dists = cdist(
            query_matrix, self._feature_db.actor_features.features, metric="cosine"
        )
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        for match_index, best_idx, min_dist in zip(
            matched_detection_indices, best_template_indices, min_dists
        ):
            confidence = self._distance_to_confidence(min_dist)
            if confidence >= MATCH_CONFIDENCE_THRESHOLD:
                name = self._feature_db.actor_features.character_names[best_idx]
                anim = self._feature_db.actor_features.animation_names[best_idx]
                matches[match_index] = Match(
                    character_id=name,
                    animation=anim,
                    confidence_score=confidence,
                )
        return matches
