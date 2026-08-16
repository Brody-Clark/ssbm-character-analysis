"""Feature extraction and template matching for actors and HUD icons."""

import logging
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from ssbmv.domain.models import (
    DetectionCandidate,
    Frame,
    HUDDetection,
    HUDState,
    Match,
    FeatureDatabase,
)
from ssbmv.core.feature_extractor import FeatureExtractor

_logger = logging.getLogger(__name__)

ACTOR_MATCH_CONFIDENCE_THRESHOLD = 0.60
HUD_MATCH_CONFIDENCE_THRESHOLD = 0.52

class Matcher:
    """Build feature templates and match detections against them."""

    def __init__(self, feature_extractor: FeatureExtractor, feature_database: FeatureDatabase):
        self._extractor = feature_extractor
        self._feature_db = feature_database

    def _distance_to_confidence(self, dist: float) -> float:
        """Convert a cosine distance into a confidence score."""
        return max(0.0, 1.0 - dist)

    def match_huds(self, frame: Frame, huds: list[HUDDetection]) -> list[HUDState]:
        """Match HUD detections with stored HUD templates."""
        query_features = []
        matched_hud_indices: list[int] = []
        matched_huds: list[HUDState] = [None] * 4

        for idx, hud in enumerate(huds):
            x, y, w, h = hud.hud_rect
            masked_img = frame.image[y : y + h, x : x + w]
            masked_img = cv.bitwise_and(masked_img, masked_img, mask=hud.binary_mask)
            success, features = self._extractor.get_hud_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_hud_indices.append(idx)

        if not query_features:
            return matched_huds

        query_matrix = np.array(query_features, dtype=np.float32)
        dists = cdist(query_matrix, self._feature_db.hud_features.features, metric="cosine")
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        for hud_index, best_idx, min_dist in zip(
            matched_hud_indices, best_template_indices, min_dists
        ):
            score = self._distance_to_confidence(min_dist)
            icon_character_id = (
                self._feature_db.hud_features.labels[best_idx]
                if score > HUD_MATCH_CONFIDENCE_THRESHOLD
                else "Unknown"
            )
            hud = huds[hud_index]
            matched_huds[hud_index] = HUDState(
                player_slot=hud.player_slot,
                icon_character_id=icon_character_id,
                hud_rect=hud.hud_rect,
            )

        return matched_huds

    def match_actors(
        self, frame: Frame, tracked_detections: list[DetectionCandidate]
    ) -> list[Match | None]:
        """Match tracked actor detections against the precompiled character templates."""
        masked_imgs = []
        image_indices = []
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
            masked_imgs.append(masked_img)
            image_indices.append(idx)

        # Get Character name
        matches: list[Match | None] = [None] * len(tracked_detections)
        matched_detection_indices = []
        query_features = []
        for idx, masked_img in zip(image_indices, masked_imgs):
            success, features = self._extractor.get_character_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_detection_indices.append(idx)

        if not query_features:
            return matches

        query_matrix = np.array(query_features, dtype=np.float32)
        dists = cdist(query_matrix, self._feature_db.character_features.features, metric="cosine")
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)
        assigned_match_img_indices = []
        for match_index, best_idx, min_dist in zip(
            matched_detection_indices, best_template_indices, min_dists
        ):
            confidence = self._distance_to_confidence(min_dist)
            if confidence >= ACTOR_MATCH_CONFIDENCE_THRESHOLD:
                name = self._feature_db.character_features.labels[best_idx]
                matches[match_index] = Match(
                    character_id=name,
                    confidence_score=confidence,
                )
                assigned_match_img_indices.append(match_index) # Only get animation features for valid matches

        # Get animations
        for idx in assigned_match_img_indices:
            masked_img = masked_imgs[idx]
            match = matches[idx]
            success, features = self._extractor.get_animation_features(image=masked_img)
            if not success:
                continue

            query_matrix = np.array([features], dtype=np.float32)
            name = match.character_id
            animation_features = self._feature_db.animation_features.get(name)
            if animation_features is None:
                continue
            dists = cdist(query_matrix, animation_features.features, metric="cosine")
            # min_dists = np.min(dists, axis=1) # TODO: confidence check?
            best_template_index = np.argmin(dists, axis=1)[0]
            match.animation_id = animation_features.labels[best_template_index]
            matches[idx] = match # TODO: needed?
        return matches
