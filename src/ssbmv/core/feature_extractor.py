import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern


class FeatureExtractor:

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a feature vector to unit length when possible."""
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-7 else vector / norm

    def _extract_saturation_histogram(self, image: cv.typing.MatLike) -> np.ndarray:
        """Extract a saturation histogram feature vector."""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = (cv.cvtColor(image, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        hist = cv.calcHist([hsv], [1], mask, [10], [0, 256]).flatten()
        return hist / (hist.sum() + 1e-7) if hist.sum() > 0 else hist

    def get_features(
        self, image: cv.typing.MatLike
    ) -> tuple[bool, np.typing.NDArray[np.float32]]:
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
        hist_hsv_sat = self._extract_saturation_histogram(image)
        _, _, w, h = cv.boundingRect(contour)
        hu_feats = self._extract_hu_moments_feature(contour=contour)
        # keys, orb_features = self._extract_orb_features(crop_img=image)

        # hsv_hist = self._extract_hsv_histogram_feature(image)

        features = np.hstack(
            (
                # self._normalize(hsv_hist),
                self._normalize(lbp_hist),
                self._normalize(desc),
                self._normalize(hist_hsv_sat),
                # self._normalize(hu_feats)
            )
        ).astype(np.float32)
        return True, features

    def _extract_orb_features(
    self, 
    crop_img: cv.typing.MatLike,
    nfeatures: int = 500,
    ):
        """Extract ORB keypoints and binary descriptors from a masked character image.

        Args:
            crop_img: Input BGR image crop with a black background ([0, 0, 0]).
            nfeatures: Maximum number of keypoints to retain.

        Returns:
            A tuple of (keypoints, descriptors).
            - keypoints: Tuple of cv.KeyPoint objects.
            - descriptors: 2D uint8 numpy array of shape (N, 32), or None if no keypoints found.
        """
        # 1. Convert to grayscale for feature detection
        gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)

        # 2. Create foreground mask (non-black pixels)
        crop_array=np.asarray(crop_img)
        mask = (np.any(crop_img != 0, axis=2)).astype(np.uint8) * 255

        # Short-circuit if mask is completely empty
        if cv.countNonZero(mask) == 0:
            return (), None

        # 3. Crop tightly to the character bounding box
        # ORB pyramid scaling works significantly better without surrounding black padding
        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        gray_tight = gray[y_min : y_max + 1, x_min : x_max + 1]
        mask_tight = mask[y_min : y_max + 1, x_min : x_max + 1]

        # 4. Initialize ORB Detector
        # fastThreshold is slightly lowered to capture fine features on smooth retro sprites
        orb = cv.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10,
        )

        # 5. Detect keypoints and compute descriptors within the masked foreground
        keypoints, descriptors = orb.detectAndCompute(gray_tight, mask=mask_tight)

        # Adjust keypoint coordinates back to the original un-cropped crop_img space
        if keypoints:
            for kp in keypoints:
                pt = list(kp.pt)
                pt[0] += x_min
                pt[1] += y_min
                kp.pt = tuple(pt)

        return keypoints, descriptors

    def _extract_hu_moments_feature(self, contour: cv.typing.MatLike):
        """Extract scale, translation, and rotation invariant shape features from a contour."""
        # 1. Calculate spatial and central moments
        moments = cv.moments(contour)

        # 2. Calculate 7 Hu Invariant Moments
        hu_moments = cv.HuMoments(moments).flatten()

        # 3. Log-transform to bring values into a comparable numerical scale
        # Hu moments span vastly different orders of magnitude (e.g. 1e-3 to 1e-15)
        for i in range(7):
            if hu_moments[i] != 0:
                hu_moments[i] = -1.0 * np.copysign(1.0, hu_moments[i]) * np.log10(np.abs(hu_moments[i]))

        return hu_moments.astype(np.float32)

    def _extract_hsv_histogram_feature(
        self,
        crop_img: cv.typing.MatLike,
        bins: tuple[int, int, int] = (4, 2, 2),
    ):
        """Extract a 1D normalized HSV histogram feature vector from a masked BGR sprite.

        Args:
            crop_img: Input BGR image crop with a black background.
            bins: Bin counts for (Hue, Saturation, Value). Default is (8, 4, 4) -> 128 features.

        Returns:
            1D float32 normalized feature vector summing to 1.0 (or all zeros if empty).
        """
        hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)

        img_array = np.asarray(crop_img)
        mask = (np.any(img_array != 0, axis=2)).astype(np.uint8) * 255
        if cv.countNonZero(mask) == 0:
            total_bins = bins[0] * bins[1] * bins[2]
            return np.zeros(total_bins, dtype=np.float32)

        # Ranges: Hue in [0, 180), Saturation in [0, 256), Value in [0, 256)
        hist = cv.calcHist(
            images=[hsv],
            channels=[0, 1, 2],
            mask=mask,
            histSize=list(bins),
            ranges=[0, 180, 0, 256, 0, 256],
        )

        feature_vector = hist.flatten()

        # L1-normalize so feature values sum to 1.0 (invariant to image/crop size)
        total_count = feature_vector.sum()
        if total_count > 0:
            feature_vector /= total_count

        return feature_vector.astype(np.float32)

    def _extract_color_structure_feature(self, crop_img, k=5, max_pixels=1000):
        # 1. Resize crop_img to cap max pixel count before HSV conversion
        # Downscaling to e.g. max 100x100 preserves color ratio while cutting 90%+ pixels
        h, w = crop_img.shape[:2]
        if h * w > 10000:
            scale = (10000 / (h * w)) ** 0.5
            crop_img = cv.resize(
                crop_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_NEAREST
            )

        hsv = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
        mask = np.any(crop_img != 0, axis=2)
        valid_pixels = hsv[mask].astype(np.float32)

        if len(valid_pixels) < k:
            return np.zeros(k, dtype=np.float32)

        # 2. Subsample valid pixels randomly if still above threshold
        if len(valid_pixels) > max_pixels:
            idx = np.random.choice(len(valid_pixels), max_pixels, replace=False)
            valid_pixels = valid_pixels[idx]

        # 3. K-Means with 1 attempt instead of 10
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv.kmeans(
            valid_pixels, k, None, criteria, attempts=1, flags=cv.KMEANS_PP_CENTERS
        )

        _, counts = np.unique(labels, return_counts=True)
        proportions = np.zeros(k, dtype=np.float32)
        sorted_counts = np.sort(counts / counts.sum())[::-1]
        proportions[: len(sorted_counts)] = sorted_counts

        return proportions

    def _extract_character_lbp(
        self, img_gray: cv.typing.MatLike, P: int = 4, R: int = 2
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
