import logging
import itertools
import numpy as np
import cv2 as cv
from skimage.feature.texture import local_binary_pattern
from scipy.spatial.distance import cdist
from ssbmv.domain.sprite_database import SpriteDatabase, CHARACTER_NAMES
from ssbmv.domain.models import GameState, TrackedActor, Frame

_logger = logging.getLogger(__name__)


class Matcher:
    def __init__(self, sprite_database: SpriteDatabase):
        templates: dict[str, list[list[np.float32]]] = self._load_templates(
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
            flat_characters 
        )

    def _get_features(self, image: cv.typing.MatLike):
        contour = self._get_shape_contour(image=image)
        if contour is None:
            return False, []
        distances = self._compute_centroid_distances(contour=contour)
        if distances is None:
            return False, []
        desc = self._get_fourier_descriptors(
            distances=distances, num_descriptors=32, sample_size=256
        )
        hist = self._extract_character_lbp(image)

        # Normalize
        hist = hist / (np.linalg.norm(hist) + 1e-7)
        desc = desc / (np.linalg.norm(desc) + 1e-7)
        features = np.hstack(hist, desc) 
        return True, features

    def _extract_character_lbp(
        self, roi_masked_bgr: cv.typing.MatLike, P: int = 8, R: int = 1
    ):
        gray = cv.cvtColor(roi_masked_bgr, cv.COLOR_BGR2GRAY)
        character_mask = (gray > 0).astype(np.uint8)
        lbp = local_binary_pattern(gray, P, R, method="uniform").astype(np.float32)

        n_bins = P + 2
        hist = cv.calcHist([lbp], [0], character_mask, [n_bins], [0, n_bins])

        # Flatten the output and normalize
        hist = hist.flatten()
        hist /= hist.sum() + 1e-7

        return hist

    def _load_templates(
        self, sprite_database: SpriteDatabase
    ) -> dict[str, list[list[np.float32]]]:
        templates = {}
        for char, sprite_sheet in sprite_database.character_sprite_db.items():
            features = []
            for s in sprite_sheet.sprite_imgs:
                success, feats = self._get_features(image=s)
                if not success:
                    _logger.error(f"Failed to extract contour from image")
                    continue
                features.append(feats)
            templates[char] = features
        return templates

    def _get_shape_contour(self, image: cv.typing.MatLike):
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
        N = len(distances)
        if N < 2:
            return None
        # Resample distances to a fixed size for uniform FFT size
        xp = np.linspace(0, N - 1, N)
        x = np.linspace(0, N - 1, sample_size)
        resampled_distances = np.interp(x, xp, distances)

        # Compute FFT
        fft_coeffs = np.fft.fft(resampled_distances)

        # Take magnitude to achieve rotation invariance (discards phase shifts)
        fft_lengths = np.abs(fft_coeffs)

        # Normalize by the DC component (fft_lengths[0]) to achieve scale invariance.
        if fft_lengths[0] < 1e-7:
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

    def _distance_to_confidence(self, dist: float, scale: float = 1.0) -> float:
        return np.exp(-scale * dist)

    def match(self, frame: Frame, game_state: GameState) -> list[TrackedActor]:
        if not game_state.active_tracks:
            return []
        
        query_features = []
        active_tracks = game_state.active_tracks
        for actor in active_tracks:
            if actor.is_active:
                x, y, w, h = actor.current_rect
                cropped_image = frame.image[y : y + h, x : x + w]
                success, features = self._get_features(image=cropped_image)
                if success:
                    query_features.append(features)

        if len(query_features) == 0:
            return active_tracks
        # Stack into a 2D matrix of shape (num_tracked_objs, feature_dimension)
        query_matrix = np.array(query_features, dtype=np.float32)

        # Shape of dists: (num_tracked_objs, total_templates)
        dists = cdist(query_matrix, self.template_matrix, metric="euclidean")

        min_dists = np.min(dists, axis=1)

        # Find the column index of the minimum distance for each tracked object
        best_template_indices = np.argmin(dists, axis=1)

        # Map the indices back to characters and update objects
        for obj, best_idx, min_dist in zip(active_tracks, best_template_indices, min_dists):
            obj.confirmed_character = self.template_characters[best_idx]
            obj.confidence_score = self._distance_to_confidence(min_dist) 

        return active_tracks
