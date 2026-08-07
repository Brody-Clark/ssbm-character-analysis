import logging
import cv2 as cv
import numpy as np
from ssbmv.domain.models import (
    Frame,
    Dimension2D,
    GameState,
    DetectionCandidate,
    HUDDetection,
)
from ssbmv.pipeline.stage_hsv_filters import STAGE_HSV_FITLERS

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

    def __init__(self, stage_name: str):
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._mog = cv.createBackgroundSubtractorMOG2(
            history=200, varThreshold=20, detectShadows=False
        )
        self._min_img = None
        self._max_img = None
        self._static_mask = None
        self._prev_frame = None

        self._hsv_filters = STAGE_HSV_FITLERS.get(stage_name, None)
        if self._hsv_filters is None:
            raise RuntimeError(f"Unsupported stage: {stage_name}")
       
    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        for item in self._hsv_filters:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)            
            mask = cv.inRange(img_hsv, lower, upper)
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
        return ~static_ui_mask # Invert to mask out dynamic pixels

    def _get_min_area(self, dim: Dimension2D) -> int:
        return int(_MIN_SPRITE_AREA_RATIO * dim.w * dim.h)

    def _extract_clean_actors(
        self,
        motion_mask: np.ndarray,
        cleaned_hsv_mask: np.ndarray,
        raw_hsv_mask: np.ndarray,
        min_actor_area: int = 500,
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
        x, y, w, h = region
        img = frame.image[y : y + h, x : x + w]

        motion_mask = self._mog.apply(frame.image)
        fg_mask = motion_mask[y : y + h, : x + w]
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
        final_actor_mask, rects = self._extract_clean_actors(
            actor_candidate_mask,
            hsv_mask_cleaned,
            hsv_mask_raw,
            min_actor_area=_MIN_SPRITE_AREA_PIXELS,
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

        candidate_mask = self._get_candidate_mask(frame=frame, region=rect)
        contours, _ = cv.findContours(
            candidate_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        candidates: list[DetectionCandidate] = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            bbox_area = w * h
            density = area / bbox_area
            aspect_ratio = float(w) / h
            if area > _MIN_SPRITE_AREA_PIXELS and (
                aspect_ratio <= _MAX_SPRITE_ASPECT_RATIO
                and aspect_ratio >= _MIN_SPRITE_ASPECT_RATIO
            ):
                if area < _MAX_SPRITE_AREA_PIXELS and density > _MIN_SPRITE_DENSITY:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv.drawContours(
                        mask, [cnt], -1, color=_COLOR_WHITE, thickness=cv.FILLED, offset=(-x,-y)
                    )
                    candidates.append(
                        DetectionCandidate(
                            rect=[x,y,w,h],
                            contour=cnt,
                            binary_mask=mask,
                        )
                    )

                else:
                    continue

        return candidates

    def _get_character_HUDs(self, frame: cv.typing.MatLike) -> list[HUDDetection]:
        """Returns the HUDStates for character HUD icons"""
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        static_mask = self._update_static_mask(frame_gray)
        # Cut out the slice containing the UI elements, ignoring the letterboxing
        # for 1280x720 video
        hud_mask = ~static_mask[536:586, 230:1080]

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

        huds = self._get_character_HUDs(frame=frame.image)
    
        # TODO: combine prediction-based local search with global search?
        # local search just uses HSV mask instead of motion.

        dimensions = frame.dimensions
        detections = self._get_candidates(
            frame=frame,
            rect=[0, 0, dimensions.w, dimensions.h],
        )

        return (detections, huds)
