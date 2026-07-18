import logging
import itertools
import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import cdist
from ssbmv.domain.sprite_database import SpriteDatabase, Character
from ssbmv.domain.models import TrackedObjectState

_logger = logging.getLogger(__name__)


class Matcher:
    def __init__(self, sprite_database: SpriteDatabase):
        templates: dict[Character, list[list[np.float32]]] = self._load_templates(
            sprite_database
        )
        flat_templates = []
        flat_characters = []
        for character, sprites in templates.items():
            for feature_vector in sprites:
                flat_templates.append(feature_vector)
                flat_characters.append(character)

        # Shape: (Total number of templates across all characters, feature_dimension)
        self.template_matrix = np.array(flat_templates, dtype=np.float32)
        self.template_characters = (
            flat_characters  # Flat list of characters for fast index lookup
        )

    def _get_features(self, image: cv.typing.MatLike):
        contour = self._get_shape_contour(image=image)
        distances = self._compute_centroid_distances(contour=contour)
        desc = self._get_fourier_descriptors(
            distances=distances, num_descriptors=32, sample_size=256
        )
        hist = self._extract_character_lbp(image)
        features = itertools.chain(hist, desc)  # TODO: normalize
        return features

    def _extract_character_lbp(
        self, roi_masked_bgr: cv.typing.MatLike, P: int = 8, R: int = 1
    ):
        gray = cv.cvtColor(roi_masked_bgr, cv.COLOR_BGR2GRAY)
        character_mask = (gray > 0).astype(np.uint8)
        lbp = local_binary_pattern(gray, P, R, method="uniform").astype(np.float32)

        n_bins = int(lbp.max() + 1)
        hist = cv.calcHist([lbp], [0], character_mask, [n_bins], [0, n_bins])

        # Flatten the output and normalize
        hist = hist.flatten()
        hist /= hist.sum() + 1e-7

        return hist

    def _load_templates(
        self, sprite_database: SpriteDatabase
    ) -> dict[Character, list[list[np.float32]]]:
        templates = {}
        for char, sprite_sheet in sprite_database.character_sprite_db.items():
            features = []
            for s in sprite_sheet.sprite_img:
                features.append(self._get_features(image=s))
            templates[char] = features
        return templates

    def _get_shape_contour(self, image: cv.typing.MatLike):
        _, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not contours:
            return None

        # Select the largest contour assuming it's the sprite
        contour = max(contours, key=cv.contourArea)
        return contour[:, 0, :]  # Reshape to (N, 2) for easier coordinate math

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
        # Resample distances to a fixed size for uniform FFT size
        # This grants sampling/scale invariance
        xp = np.linspace(0, len(distances), len(distances))
        x = np.linspace(0, len(distances), sample_size)
        resampled_distances = np.interp(x, xp, distances)

        # Compute FFT
        fft_coeffs = np.fft.fft(resampled_distances)

        # Take magnitude to achieve rotation invariance (discards phase shifts)
        fft_lengths = np.abs(fft_coeffs)

        # Normalize by the DC component (fft_lengths[0]) to achieve scale invariance.
        # Skip the DC component in the final vector as it's now always 1.0
        if fft_lengths[0] == 0:
            return None
        normalized_fds = fft_lengths[1 : num_descriptors + 1] / fft_lengths[0]

        return normalized_fds

    def _get_best_match(self, roi_features, template_dict, threshold=0.15):
        best_match = "Unknown"
        min_distance = float("inf")

        for character_name, template_features in template_dict.items():
            # Calculate Euclidean Distance between the two feature vectors
            distance = np.linalg.norm(roi_features - template_features)

            if distance < min_distance:
                min_distance = distance
                best_match = character_name

        # Reject bad matches
        if min_distance > threshold:
            return "Unknown"

        return best_match

    def match(self, tracked_objs: list[TrackedObjectState]) -> list[TrackedObjectState]:
        if not tracked_objs:
            return tracked_objs
        query_features = [
            self._get_features(image=obj.region.masked_rgb_slice)
            for obj in tracked_objs
        ]

        # Stack into a 2D matrix of shape (num_tracked_objs, feature_dimension)
        query_matrix = np.array(query_features, dtype=np.float32)

        # Shape of dists: (num_tracked_objs, total_templates)
        dists = cdist(query_matrix, self.template_matrix, metric="euclidean")

        # Find the column index of the minimum distance for each tracked object
        best_template_indices = np.argmin(dists, axis=1)

        # Map the indices back to characters and update objects
        for obj, best_idx in zip(tracked_objs, best_template_indices):
            obj.sprite_name = self.template_characters[best_idx]

        return tracked_objs
