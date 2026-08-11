"""Feature extraction and template matching for actors and HUD icons."""

import logging
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
from ssbmv.domain.models import DetectionCandidate, Frame, HUDDetection, HUDState, Match
from ssbmv.domain.sprite_database import SpriteDatabase

_logger = logging.getLogger(__name__)

MATCH_CONFIDENCE_THRESHOLD = 0.50


class Matcher:
    """Build feature templates and match detections against them."""

    def __init__(self, sprite_database: SpriteDatabase):
        templates = self._load_character_template(sprite_database)
        flat_templates, flat_characters = [], []
        for character_anim, features in templates.items():
            flat_templates.extend(features)
            flat_characters.extend([character_anim] * len(features))

        self._character_template_matrix = np.array(flat_templates, dtype=np.float32)
        self._template_characters = flat_characters

        hud_templates = self._load_character_hud_template(sprite_database)
        flat_hud_templates, flat_hud_characters = [], []
        for character, hud_features in hud_templates.items():
            flat_hud_templates.append(hud_features)
            flat_hud_characters.append(character)

        self._hud_template_matrix = np.array(flat_hud_templates, dtype=np.float32)
        self._hud_characters = flat_hud_characters

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a feature vector to unit length when possible."""
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-7 else vector / norm

    def _extract_color_histogram(self, image: cv.typing.MatLike) -> np.ndarray:
        """Extract a saturation histogram feature vector."""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = (cv.cvtColor(image, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        hist = cv.calcHist([hsv], [1], mask, [10], [0, 256]).flatten()
        return hist / (hist.sum() + 1e-7) if hist.sum() > 0 else hist

    def _get_features(self, image: cv.typing.MatLike):
        """Extract a combined feature vector for a candidate sprite image."""
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        contour = self._get_shape_contour(img_gray=img_gray)
        if contour is None:
            return False, []

        distances = self._compute_centroid_distances(contour=contour)
        if distances is None:
            return False, []

        desc = self._get_fourier_descriptors(
            distances=distances, num_descriptors=24, sample_size=256
        )
        lbp_hist = self._extract_character_lbp(img_gray)
        hist_hsv_sat = self._extract_color_histogram(image)
        _, _, w, h = cv.boundingRect(contour)
        aspect_ratio = float(w) / h
        aspect_feat = np.array([np.log(aspect_ratio + 1e-5) * 0.2], dtype=np.float32)

        features = np.hstack(
            (
                self._normalize(lbp_hist),
                self._normalize(desc),
                self._normalize(hist_hsv_sat),
                aspect_feat,
            )
        ).astype(np.float32)
        return True, features

    def _extract_character_lbp(
        self, img_gray: cv.typing.MatLike, P: int = 8, R: int = 2
    ):
        """Compute a normalized local binary pattern histogram for the input image."""
        character_mask = (img_gray > 0).astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
        valid_lbp_mask = cv.erode(character_mask, kernel)
        if np.count_nonzero(valid_lbp_mask) == 0:
            valid_lbp_mask = character_mask

        lbp = local_binary_pattern(img_gray, P, R, method="uniform").astype(np.float32)
        hist = cv.calcHist([lbp], [0], valid_lbp_mask, [P + 2], [0, P + 2]).flatten()
        return hist / (hist.sum() + 1e-7)

    def _load_character_template(
        self, sprite_database: SpriteDatabase
    ) -> dict[str, list[list[np.float32]]]:
        """Extract feature templates from the character sprite sheets."""
        templates = {}
        for char, sprite_sheet in sprite_database.character_sprite_db.items():
            for idx, sprite_img in enumerate(sprite_sheet.sprite_imgs):
                success, feats = self._get_features(image=sprite_img)
                if not success:
                    _logger.error("Failed to extract features from image.")
                    continue

                anim = sprite_sheet.sprite_names[idx]
                key = f"{char}_{anim}"
                templates.setdefault(key, []).append(feats)
        return templates

    def _load_character_hud_template(
        self, sprite_database: SpriteDatabase
    ) -> dict[str, list[np.float32]]:
        """Extract feature templates from the HUD sprite images."""
        templates = {}
        for char, hud_img in sprite_database.character_hud_db.items():
            success, feats = self._get_features(image=hud_img)
            if not success:
                _logger.error("Failed to extract features from image")
                continue
            templates[char] = feats
        return templates

    def _get_shape_contour(self, img_gray: cv.typing.MatLike):
        """Find the largest external contour in the grayscale image."""
        _, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return None if not contours else max(contours, key=cv.contourArea)[:, 0, :]

    def _compute_centroid_distances(self, contour: cv.typing.MatLike):
        """Compute the distance from each contour point to the contour centroid."""
        moments = cv.moments(contour.astype(np.float32))
        if moments["m00"] == 0:
            return None

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)

    def _get_fourier_descriptors(
        self, distances, num_descriptors: int = 32, sample_size: int = 256
    ):
        """Resample the contour distances and return a compact Fourier descriptor vector."""
        if len(distances) < 2:
            return None

        xp = np.linspace(0, len(distances) - 1, len(distances))
        x = np.linspace(0, len(distances) - 1, sample_size)
        resampled_distances = np.interp(x, xp, distances)

        fft_coeffs = np.fft.fft(resampled_distances)
        dc_component = np.abs(fft_coeffs[0])
        if dc_component < 1e-7:
            return None

        coeffs = fft_coeffs[1 : num_descriptors + 1]
        return np.concatenate([np.real(coeffs), np.imag(coeffs)]) / dc_component

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
            success, features = self._get_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_hud_indices.append(idx)

        if not query_features:
            return matched_huds

        query_matrix = np.array(query_features, dtype=np.float32)
        dists = cdist(query_matrix, self._hud_template_matrix, metric="cosine")
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        for hud_index, best_idx, min_dist in zip(
            matched_hud_indices, best_template_indices, min_dists
        ):
            score = self._distance_to_confidence(min_dist)
            icon_character_id = (
                self._hud_characters[best_idx]
                if score > MATCH_CONFIDENCE_THRESHOLD
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
            success, features = self._get_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_detection_indices.append(idx)

        if not query_features:
            return matches

        query_matrix = np.array(query_features, dtype=np.float32)
        dists = cdist(query_matrix, self._character_template_matrix, metric="cosine")
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        for match_index, best_idx, min_dist in zip(
            matched_detection_indices, best_template_indices, min_dists
        ):
            confidence = self._distance_to_confidence(min_dist)
            if confidence >= MATCH_CONFIDENCE_THRESHOLD:
                matches[match_index] = Match(
                    character_id=self._template_characters[best_idx],
                    confidence_score=confidence,
                )

        return matches
