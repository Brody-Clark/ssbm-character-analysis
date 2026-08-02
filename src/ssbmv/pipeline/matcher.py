import logging
import numpy as np
import cv2 as cv
from skimage.feature.texture import local_binary_pattern
from scipy.spatial.distance import cdist
from ssbmv.domain.sprite_database import SpriteDatabase, SpriteSheet
from ssbmv.domain.models import GameState, TrackedActor, Frame, HUDState

_logger = logging.getLogger(__name__)


class Matcher:
    def __init__(self, sprite_database: SpriteDatabase):
        templates: dict[str, list[list[np.float32]]] = self._load_template(sprite_database.character_sprite_db)
        flat_templates = []
        flat_characters = []
        for character, sprites in templates.items():
            for feature_vector in sprites:
                flat_templates.append(feature_vector)
                flat_characters.append(character)

        # Shape: (Total number of templates across all characters, feature_dimension)
        self._character_template_matrix = np.array(flat_templates, dtype=np.float32)
        self._template_characters = flat_characters

        templates = self._load_template(sprite_database.character_hud_db)

    def _extract_color_histogram(self, image: cv.typing.MatLike) -> np.ndarray:
        """Exract 1D saturation feature vector"""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = (cv.cvtColor(image, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # 4 bins for Saturation
        hist = cv.calcHist([hsv], [1], mask, [8], [0, 256])
        hist = hist.flatten()

        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        return hist

    def _get_features(self, image: cv.typing.MatLike):
        # Fourier descriptors
        contour = self._get_shape_contour(image=image)
        if contour is None:
            return False, []
        distances = self._compute_centroid_distances(contour=contour)
        if distances is None:
            return False, []
        desc = self._get_fourier_descriptors(
            distances=distances, num_descriptors=32, sample_size=256
        )

        # LBP
        hist = self._extract_character_lbp(image)

        # HSV Saturation
        hist_hsv_sat = self._extract_color_histogram(image)

        # Normalize
        hist = hist / (np.linalg.norm(hist) + 1e-7)
        desc = desc / (np.linalg.norm(desc) + 1e-7)
        color_norm = hist_hsv_sat / (np.linalg.norm(hist_hsv_sat) + 1e-7)

        features = np.hstack((hist, desc, color_norm)).astype(np.float32)
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

    def _load_template(self, sprite_database: dict[str, SpriteSheet])-> dict[str, list[list[np.float32]]]:
        templates = {}
        for char, sprite_sheet in sprite_database.items():
            features = []
            for s in sprite_sheet.sprite_imgs:
                success, feats = self._get_features(image=s)
                if not success:
                    _logger.error(f"Failed to extract features from image")
                    continue
                features.append(feats)
            templates[char] = features
        return templates
    
    def _get_shape_contour(self, image: cv.typing.MatLike):
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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
        return np.exp(-scale * dist)

    def _match_huds(self, frame: Frame, huds: list[HUDState]) -> tuple[bool, int]:
        """Matches hud elements with templates. Returns a tuple containig success flag and number of matched icons"""

        query_features = []
        matched_huds: list[HUDState] = []
        for hud in huds:
            x, y, w, h = hud.hud_rect
            masked_img = frame.image[y : y + h, x : x + w]
            masked_img = cv.bitwise_and(masked_img, masked_img, mask=hud.binary_mask)
            success, features = self._get_features(image=masked_img)
            if success:
                query_features.append(features)
                matched_huds.append(hud)

        if not query_features:
            return (False,0)

        query_matrix = np.array(query_features, dtype=np.float32)

        # Compute pairwise Euclidean distances
        dists = cdist(query_matrix, self._character_hud_matrix, metric="euclidean")
        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        count = 0
        for hud, best_idx, min_dist in zip(
            matched_huds, best_template_indices, min_dists
        ):
            score = self._distance_to_confidence(min_dist)
            if score > 0.5:
                hud.icon_character_id = self._template_character_huds[best_idx]
                count+=1

        return (True, count)
    
    def match(self, frame: Frame, game_state: GameState) -> list[TrackedActor]:
        if not game_state.huds_found:
            self._match_huds(game_state.hud_states)
        if not game_state.active_tracks:
            return []

        matched_actors = []
        query_features = []

        # Keep actors and features aligned by tracking valid indices
        for actor in game_state.active_tracks:
            if actor.is_active:
                x, y, w, h = actor.current_rect
                cropped_image = frame.image[y : y + h, x : x + w]

                if cropped_image.size == 0:
                    continue

                success, features = self._get_features(image=cropped_image)
                if success:
                    query_features.append(features)
                    matched_actors.append(actor)

        if not query_features:
            return game_state.active_tracks

        query_matrix = np.array(query_features, dtype=np.float32)

        # Compute pairwise Euclidean distances
        dists = cdist(query_matrix, self._character_template_matrix, metric="euclidean")

        min_dists = np.min(dists, axis=1)
        best_template_indices = np.argmin(dists, axis=1)

        for actor, best_idx, min_dist in zip(
            matched_actors, best_template_indices, min_dists
        ):
            actor.confirmed_character = self._template_characters[best_idx]
            actor.confidence_score = self._distance_to_confidence(min_dist)

        return game_state.active_tracks
