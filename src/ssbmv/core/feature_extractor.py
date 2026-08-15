import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern


class FeatureExtractor:
    def __init__(self):
        pass

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
    
    def get_features(
        self, image: cv.typing.MatLike
    ) -> tuple[bool, np.ndarray[np._AnyShape, np.dtype[np.floating[np._32Bit]]]]:
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

        color_struct = self._extract_color_structure_feature(image)

        features = np.hstack(
            (
                self._normalize(lbp_hist),
                self._normalize(desc),
                self._normalize(hist_hsv_sat),
                self._normalize(color_struct),
                aspect_feat,
            )
        ).astype(np.float32)
        return True, features

    def _extract_color_structure_feature(self, crop_img, k=5):
        hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # 2. Filter out transparent/background pixels if masked
        # pixels = pixels[mask > 0]

        # K-Means clustering to find major color proportions
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv.kmeans(
            pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        )

        # Count proportions of each cluster & sort descending
        _, counts = np.unique(labels, return_counts=True)
        proportions = np.sort(counts / counts.sum())[::-1]

        return proportions

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
