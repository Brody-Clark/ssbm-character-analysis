import logging
import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern, hog
from scipy.spatial.distance import cdist
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.domain.models import (
    Frame,
    HUDState,
    HUDDetection,
    DetectionCandidate,
    Match,
)

_logger = logging.getLogger(__name__)


class Matcher:
    def __init__(self, sprite_database: SpriteDatabase):
        # Load Character sprite templates
        self._flat_templates = []
        templates: dict[str, list[list[np.float32]]] = self._load_character_template(
            sprite_database
        )
        self._temps = sprite_database.character_sprite_db
        flat_templates = []
        flat_characters = []
        for character, sprites in templates.items():
            for feature_vector in sprites:
                flat_templates.append(feature_vector)
                flat_characters.append(character)

        # Shape: (Total number of templates across all characters, feature_dimension)
        self._character_template_matrix = np.array(flat_templates, dtype=np.float32)
        self._template_characters = flat_characters

        # Load HUD templates
        hud_templates = self._load_character_hud_template(sprite_database)
        flat_hud_templates = []
        flat_hud_characters = []
        for character, hud_features in hud_templates.items():
            flat_hud_templates.append(hud_features)
            flat_hud_characters.append(character)
        self._hud_template_matrix = np.array(flat_hud_templates, dtype=np.float32)
        self._hud_characters = flat_hud_characters

    def _extract_color_histogram(self, image: cv.typing.MatLike) -> np.ndarray:
        """Exract 1D saturation feature vector"""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = (cv.cvtColor(image, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # Saturation
        hist = cv.calcHist([hsv], [1], mask, [10], [0, 256])
        hist = hist.flatten()

        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        return hist

    def _get_features(self, image: cv.typing.MatLike):
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Fourier descriptors
        contour = self._get_shape_contour(image=img_gray)
        if contour is None:
            return False, []
        distances = self._compute_centroid_distances(contour=contour)
        if distances is None:
            return False, []
        desc = self._get_fourier_descriptors(
            distances=distances, num_descriptors=42, sample_size=256
        )

        # LBP
        hist = self._extract_character_lbp(img_gray)

        # HSV Saturation
        hist_hsv_sat = self._extract_color_histogram(image)

        # HOG
        # hog_feats, _ = hog(
        #     image,
        #     orientations=6,
        #     pixels_per_cell=(16, 16),
        #     cells_per_block=(1, 1),
        #     visualize=True,
        #     channel_axis=-1,
        # )

        # Normalize
        hist = hist / (np.linalg.norm(hist) + 1e-7)
        desc = desc / (np.linalg.norm(desc) + 1e-7)
        color_norm = hist_hsv_sat / (np.linalg.norm(hist_hsv_sat) + 1e-7)
        # hog_norm = hog_feats / (np.linalg.norm(hog_feats) + 1e-7)
        # hog_norm = hog_norm[:14]  # 32 + 8 + 10 + 78 = 128
        # hog_norm = np.pad(hog_norm, (0, 14 - len(hog_norm)))
        features = np.hstack((hist, desc, color_norm)).astype(np.float32)
        return True, features

    def _extract_character_lbp(
        self, img_gray: cv.typing.MatLike, P: int = 10, R: int = 2
    ):
        character_mask = (img_gray > 0).astype(np.uint8)
        lbp = local_binary_pattern(img_gray, P, R, method="uniform").astype(np.float32)

        n_bins = P + 2
        hist = cv.calcHist([lbp], [0], character_mask, [n_bins], [0, n_bins])

        # Flatten the output and normalize
        hist = hist.flatten()
        hist /= hist.sum() + 1e-7

        return hist

    def _load_character_template(
        self, sprite_database: SpriteDatabase
    ) -> dict[str, list[list[np.float32]]]:
        sprite_sheets = sprite_database.character_sprite_db.items()
        templates = {}
        for char, sprite_sheet in sprite_sheets:
            features = []
            for s in sprite_sheet.sprite_imgs:
                self._flat_templates.append(s)
                success, feats = self._get_features(image=s)
                if not success:
                    _logger.error("Failed to extract features from image.")
                    continue
                features.append(feats)
            templates[char] = features
        return templates

    def _load_character_hud_template(
        self, sprite_database: SpriteDatabase
    ) -> dict[str, list[np.float32]]:
        hud_sheets = sprite_database.character_hud_db
        templates = {}
        for char, hud_img in hud_sheets.items():
            success, feats = self._get_features(image=hud_img)
            if not success:
                _logger.error("Failed to extract features from image")
                continue
            templates[char] = feats
        return templates

    def _get_shape_contour(self, img_gray: cv.typing.MatLike):
        _, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv.contourArea)
        return contour[:, 0, :]

    def _compute_centroid_distances(self, contour: cv.typing.MatLike):
        M = cv.moments(contour.astype(np.float32))
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Compute Euclidean distances from centroid to all boundary points
        distances = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
        return distances

    def _get_fourier_descriptors(
        self, distances, num_descriptors: int = 32, sample_size: int = 256
    ):
        N = len(distances)
        if N < 2:
            return None
        # Resample distances to a fixed size for uniform FFT size
        xp = np.linspace(0, N - 1, N)
        x = np.linspace(0, N - 1, sample_size)
        resampled_distances = np.interp(x, xp, distances)

        # Compute FFT
        fft_coeffs = np.fft.fft(resampled_distances)
        fft_lengths = np.abs(fft_coeffs)
        if fft_lengths[0] < 1e-7:
            return None

        # Normalize by the DC component for scale invariance.
        normalized_fds = (fft_lengths[1 : num_descriptors + 1] / fft_lengths[0]).astype(
            np.float32
        )
        return normalized_fds

    def _distance_to_confidence(self, dist: float, scale: float = 1.0) -> float:
        return 1 / (1 + dist)

    def match_huds(self, frame: Frame, huds: list[HUDDetection]) -> list[HUDState]:
        """Matches HUD elements with templates."""
        query_features = []
        matched_hud_indices: list[int] = []
        for i, hud in enumerate(huds):
            x, y, w, h = hud.hud_rect
            masked_img = frame.image[y : y + h, x : x + w]
            masked_img = cv.bitwise_and(masked_img, masked_img, mask=hud.binary_mask)
            success, features = self._get_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_hud_indices.append(i)

        if not query_features:
            return

        query_matrix = np.array(query_features, dtype=np.float32)

        # Compute pairwise Euclidean distances
        dists = cdist(query_matrix, self._hud_template_matrix, metric="euclidean")
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        matched_huds: list[HUDState] = [None] * 4
        for hud_index, best_idx, min_dist in zip(
            matched_hud_indices, best_template_indices, min_dists
        ):
            score = self._distance_to_confidence(min_dist)
            if score > 0.5:
                icon_character_id = self._hud_characters[best_idx]
            else:
                icon_character_id = "Unknown"
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
        """Matches active tracked actors and HUD elements in the current frame with precompiled templates."""

        matches = [None] * len(tracked_detections)
        matched_detection_indices = []
        query_features = []
        # Keep actors and features aligned
        for i, detection in enumerate(tracked_detections):
            if detection is None:
                continue

            x, y, w, h = detection.rect
            cropped_image = frame.image[y : y + h, x : x + w]
            if cropped_image.size == 0:
                continue

            masked_img = cv.bitwise_and(
                cropped_image, cropped_image, mask=detection.binary_mask
            )
            success, features = self._get_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_detection_indices.append(i)

        if not query_features:
            return matches

        query_matrix = np.array(query_features, dtype=np.float32)

        # Compute pairwise Euclidean distances
        dists = cdist(query_matrix, self._character_template_matrix, metric="euclidean")

        # Set character id and confidence score for matched actors
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)
        for match_index, best_idx, min_dist in zip(
            matched_detection_indices, best_template_indices, min_dists
        ):
            new_match = Match(
                character_id=self._template_characters[best_idx],
                confidence_score=self._distance_to_confidence(min_dist),
            )
            matches[match_index] = new_match

        return matches
