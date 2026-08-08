import logging
import cv2 as cv
import numpy as np
from ssbmv.domain.models import (
    Frame,
    Dimension2D,
    DetectionCandidate,
    HUDDetection,
)
from ssbmv.domain.stage_hsv_filters import STAGE_HSV_FITLERS

_logger = logging.getLogger(__name__)

_STATIC_MASK_MAX_ALLOWED_PIXEL_DIFF = 25
_MIN_SPRITE_AREA_RATIO = 0.006
_MIN_SPRITE_AREA_PIXELS = 2000
_MAX_SPRITE_AREA_PIXELS = 70000
_MIN_SPRITE_ASPECT_RATIO = 0.35
_MAX_SPRITE_ASPECT_RATIO = 1.65
_MIN_SPRITE_DENSITY = 0.3
_COLOR_WHITE = 255


class Detector:
    """Detects actors in frames of SSBM gameplay."""

    def __init__(self, stage_name: str, scale_factor: float = 0.5):
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._mog = cv.createBackgroundSubtractorMOG2(
            history=200, varThreshold=20, detectShadows=False
        )

        # Static mask setup
        self._min_img = None
        self._max_img = None
        self._static_mask = None

        # Scale factors for operating on smaller image
        self._scale_factor = scale_factor
        self._scale_up_factor = 1.0 / scale_factor
        self._scaled_min_sprite_area = int(
            round(_MIN_SPRITE_AREA_PIXELS * scale_factor * scale_factor)
        )
        self._scaled_max_sprite_area = int(
            round(_MAX_SPRITE_AREA_PIXELS * scale_factor * scale_factor)
        )

        # Load HSV filters for specified stage
        self._hsv_filters = STAGE_HSV_FITLERS.get(stage_name, None)
        if self._hsv_filters is None:
            raise RuntimeError(f"Unsupported stage: {stage_name}")

        # Pre-compile HSV mask lists to reduce runtime allocation
        self._hsv_lower = np.array(
            [item["lower"] for item in self._hsv_filters], dtype=np.uint8
        )
        self._hsv_upper = np.array(
            [item["upper"] for item in self._hsv_filters], dtype=np.uint8
        )
        self._hsv_count = len(self._hsv_upper)

    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        """Removes stage geometry using HSV masking and leaves isolated character masks."""
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        for i in range(self._hsv_count):
            mask = cv.inRange(img_hsv, self._hsv_lower[i], self._hsv_upper[i])
            combined_mask = cv.bitwise_or(combined_mask, mask)
        return cv.bitwise_not(combined_mask)

    def _update_static_mask(self, frame_gray: cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Updates pixel ranges and returns static UI mask (255 = Static UI, 0 = Gameplay).
        """
        # Lazy initialization to guarantee exact shape matching on Frame 1
        if self._min_img is None or self._min_img.shape != frame_gray.shape:
            self._min_img = frame_gray.copy()
            self._max_img = frame_gray.copy()

        # Update minimum and maximum seen values at each pixel coordinate
        np.minimum(self._min_img, frame_gray, out=self._min_img)
        np.maximum(self._max_img, frame_gray, out=self._max_img)

        # Compute max variation per pixel across all observed frames
        range_img = cv.subtract(self._max_img, self._min_img)

        # Static pixels: variation <= max allowed difference
        _, static_ui_mask = cv.threshold(
            range_img, _STATIC_MASK_MAX_ALLOWED_PIXEL_DIFF, 255, cv.THRESH_BINARY_INV
        )
        return static_ui_mask

    def _get_min_area(self, dim: Dimension2D) -> int:
        return int(_MIN_SPRITE_AREA_RATIO * dim.w * dim.h)

    def _resize_image(self, img: cv.typing.MatLike, scale: float) -> cv.typing.MatLike:
        return cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)

    def _rescale_rect(self, rect: cv.typing.Rect) -> list[int]:
        return [int(round(coord * self._scale_up_factor)) for coord in rect]

    def _rescale_mask(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv.resize(mask, (width, height), interpolation=cv.INTER_NEAREST)

    def _rescale_contour(self, contour: np.ndarray) -> np.ndarray:
        scaled = contour.astype(np.float32) * self._scale_up_factor
        return np.rint(scaled).astype(np.int32)

    def _extract_clean_actors(
        self,
        motion_mask: np.ndarray,
        cleaned_hsv_mask: np.ndarray,
        raw_hsv_mask: np.ndarray,
        min_actor_area: int,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Parameters:

        - motion_mask: Mask created from combining motion + cleaned HSV mask.
        - cleaned_hsv_mask: Heavily cleaned/dilated HSV mask (has complete shapes
        but enlarged edges).
        - raw_hsv_mask: Un-dilated raw HSV mask (has crisp actor edges + background
        noise).

        Returns:
        - final_actor_mask: Slim, high-precision actor binary mask.
        - actor_rects: Merged bounding boxes around detected actors.
        """
        # Find initial fragmented contours from motion + hsv result
        # Havily dilate to close actor contours
        motion_mask = cv.dilate(motion_mask, self._edge_dilation_kernel, iterations=1)
        initial_contours, _ = cv.findContours(
            motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        raw_rects = []
        for cnt in initial_contours:
            if cv.contourArea(cnt) >= min_actor_area:
                raw_rects.append(cv.boundingRect(cnt))

        if not raw_rects:
            return np.zeros_like(raw_hsv_mask), []

        # Extract solid actor contours from the Cleaned HSV Mask
        # Create a spatial ROI mask of just the merged actor regions
        spatial_roi = np.zeros_like(cleaned_hsv_mask)
        for x, y, w, h in raw_rects:
            spatial_roi[y : y + h, x : x + w] = 255

        # Isolate cleaned HSV mask within active actor bounding boxes
        actor_cleaned_hsv = cv.bitwise_and(cleaned_hsv_mask, spatial_roi)

        # Find the complete outer contours of the actors within these regions
        actor_contours, _ = cv.findContours(
            actor_cleaned_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Fill actor contours solid to cover inner gaps/holes
        solid_actor_mask = np.zeros_like(cleaned_hsv_mask)
        cv.drawContours(
            solid_actor_mask, actor_contours, -1, color=255, thickness=cv.FILLED
        )

        # Intersect filled actor shape with RAW HSV mask
        final_actor_mask = cv.bitwise_and(raw_hsv_mask, solid_actor_mask)
        return final_actor_mask, raw_rects

    def _get_candidate_mask(
        self,
        frame: Frame,
        region: cv.typing.Rect,
    ) -> cv.typing.MatLike:
        """Combines multiple methods to create a clean candidate actor mask from the input image."""
        x, y, w, h = region
        img = frame.image[y : y + h, x : x + w]

        motion_mask = self._mog.apply(frame.image)
        fg_mask = motion_mask[y : y + h, x : x + w]
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        _, fg_mask = cv.threshold(fg_mask, 125, 255, cv.THRESH_BINARY)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv.dilate(fg_mask, kernel=kernel)

        hsv_mask_raw = self._get_hsv_mask(img=img)
        hsv_mask_cleaned = cv.morphologyEx(hsv_mask_raw, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        hsv_mask_cleaned = cv.dilate(hsv_mask_cleaned, kernel, iterations=1)
        hsv_mask_cleaned = cv.morphologyEx(
            hsv_mask_cleaned, cv.MORPH_CLOSE, kernel, iterations=1
        )

        actor_candidate_mask = hsv_mask_cleaned & fg_mask
        final_actor_mask, _ = self._extract_clean_actors(
            actor_candidate_mask,
            hsv_mask_cleaned,
            hsv_mask_raw,
            min_actor_area=self._scaled_min_sprite_area,
        )

        final_actor_mask = cv.dilate(final_actor_mask, kernel, iterations=1)
        final_actor_mask = cv.morphologyEx(
            final_actor_mask, cv.MORPH_CLOSE, kernel, iterations=1
        )
        return final_actor_mask

    def _get_candidates(
        self,
        frame: Frame,
        rect: cv.typing.Rect,
    ) -> list[DetectionCandidate]:
        """Gets candidate mask, then applies geometric thresholds to create final detection candidates."""

        candidate_mask = self._get_candidate_mask(frame=frame, region=rect)
        contours, _ = cv.findContours(
            candidate_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Apply geometric thresholds to remove small or poorly shaped detections
        candidates: list[DetectionCandidate] = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            bbox_area = w * h
            density = area / bbox_area
            aspect_ratio = float(w) / h
            if area > self._scaled_min_sprite_area and (
                aspect_ratio <= _MAX_SPRITE_ASPECT_RATIO
                and aspect_ratio >= _MIN_SPRITE_ASPECT_RATIO
            ):
                if (
                    area < self._scaled_max_sprite_area
                    and density > _MIN_SPRITE_DENSITY
                ):
                    # Fill in mask for just this candidate
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv.drawContours(
                        mask,
                        [cnt],
                        -1,
                        color=_COLOR_WHITE,
                        thickness=cv.FILLED,
                        offset=(-x, -y),
                    )
                    candidates.append(
                        DetectionCandidate(
                            rect=[x, y, w, h],
                            contour=cnt,
                            binary_mask=mask,
                        )
                    )

        return candidates

    def _get_character_HUDs(self, frame: cv.typing.MatLike) -> list[HUDDetection]:
        """Returns the HUDStates for character HUD icons using temporal masking."""
        # Cut out the slice containing the UI elements, ignoring the pillar boxes
        # for 1280x720 video
        hud_slice = frame[536:586, 230:1080]
        hud_slice_gray = cv.cvtColor(hud_slice, cv.COLOR_BGR2GRAY)
        hud_mask = self._update_static_mask(hud_slice_gray)

        # Clean up mask
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        hud_mask = cv.morphologyEx(hud_mask, cv.MORPH_OPEN, kernel, iterations=1)
        hud_mask = cv.morphologyEx(hud_mask, cv.MORPH_CLOSE, kernel, iterations=1)

        # Slice out each individual profile sprite
        candidate_huds = []
        x, y, w, h = 0, 0, 50, 50
        for _ in range(4):
            hud_state = HUDDetection(
                0,
                hud_rect=[x + 230, y + 536, w, h],
                binary_mask=hud_mask[y : y + h, x : x + w],
            )
            candidate_huds.append(hud_state)
            x += 212  # HUD element width on 1280x720 footage
        return candidate_huds

    def detect(
        self, frame: Frame
    ) -> tuple[list[DetectionCandidate], list[HUDDetection]]:
        """Locates regions of interest containing dynamic actors from a frame of SSBM gameplay."""

        # Get HUDs first with full frame
        huds = self._get_character_HUDs(frame=frame.image)

        # Downscale the image for heavy actor detection work
        scaled_image = self._resize_image(frame.image, self._scale_factor)
        scaled_dimensions = Dimension2D(
            w=int(round(frame.dimensions.w * self._scale_factor)),
            h=int(round(frame.dimensions.h * self._scale_factor)),
        )
        scaled_frame = Frame(
            frame_id=frame.frame_id,
            image=scaled_image,
            dimensions=scaled_dimensions,
            timestamp=frame.timestamp,
        )

        scaled_candidates = self._get_candidates(
            frame=scaled_frame,
            rect=[0, 0, scaled_dimensions.w, scaled_dimensions.h],
        )

        detections: list[DetectionCandidate] = []
        for candidate in scaled_candidates:
            rescaled_rect = self._rescale_rect(candidate.rect)
            rescaled_mask = self._rescale_mask(
                candidate.binary_mask, rescaled_rect[2], rescaled_rect[3]
            )
            rescaled_contour = self._rescale_contour(candidate.contour)
            detections.append(
                DetectionCandidate(
                    rect=rescaled_rect,
                    contour=rescaled_contour,
                    binary_mask=rescaled_mask,
                )
            )

        return (detections, huds)
