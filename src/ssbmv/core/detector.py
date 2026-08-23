"""Detect actor candidates and HUD regions in Super Smash Bros. Melee frames."""

import cv2 as cv
import numpy as np
from ssbmv.domain.models import DetectionCandidate, Dimension2D, Frame
from ssbmv.domain.stage_hsv_filters import STAGE_HSV_FILTERS

_STATIC_MASK_MAX_ALLOWED_PIXEL_DIFF = 64
_MIN_SPRITE_AREA_RATIO = 0.006
_MIN_SPRITE_AREA_PIXELS = 2000
_MAX_SPRITE_AREA_PIXELS = 70000
_MIN_SPRITE_ASPECT_RATIO = 0.35
_MAX_SPRITE_ASPECT_RATIO = 1.65
_MIN_SPRITE_DENSITY = 0.3
_COLOR_WHITE = 255


class Detector:
    """Detect actors and HUD elements from gameplay frames."""

    def __init__(self, stage_name: str, scale_factor: float = 0.5):
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._mog = cv.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=20,
            detectShadows=False,
        )

        self._scale_factor = scale_factor
        self._scale_up_factor = 1.0 / scale_factor
        self._scaled_min_sprite_area = int(
            round(_MIN_SPRITE_AREA_PIXELS * scale_factor * scale_factor)
        )
        self._scaled_max_sprite_area = int(
            round(_MAX_SPRITE_AREA_PIXELS * scale_factor * scale_factor)
        )

        self._hsv_filters = STAGE_HSV_FILTERS.get(stage_name)
        if self._hsv_filters is None:
            raise RuntimeError(f"Unsupported stage: {stage_name}")

        self._hsv_lower = np.array(
            [item["lower"] for item in self._hsv_filters], dtype=np.uint8
        )
        self._hsv_upper = np.array(
            [item["upper"] for item in self._hsv_filters], dtype=np.uint8
        )
        self._hsv_count = len(self._hsv_upper)

    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        """Remove stage geometry using HSV masking and leave isolated character masks."""
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        combined_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        for idx in range(self._hsv_count):
            mask = cv.inRange(img_hsv, self._hsv_lower[idx], self._hsv_upper[idx])
            combined_mask = cv.bitwise_or(combined_mask, mask)
        return cv.bitwise_not(combined_mask)

    def _resize_image(self, img: cv.typing.MatLike, scale: float) -> cv.typing.MatLike:
        """Resize an image using the configured scale factor."""
        return cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)

    def _rescale_rect(self, rect: cv.typing.Rect) -> list[int]:
        """Scale a rectangle back to the original frame size."""
        return [int(round(coord * self._scale_up_factor)) for coord in rect]

    def _rescale_mask(self, mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize a binary mask back to a full-resolution bounding box."""
        return cv.resize(mask, (width, height), interpolation=cv.INTER_NEAREST)

    def _rescale_contour(self, contour: np.ndarray) -> np.ndarray:
        """Scale a contour back to the original frame size."""
        return np.rint(contour.astype(np.float32) * self._scale_up_factor).astype(
            np.int32
        )

    def _extract_clean_actors(
        self,
        motion_mask: np.ndarray,
        cleaned_hsv_mask: np.ndarray,
        raw_hsv_mask: np.ndarray,
        min_actor_area: int,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Use motion and HSV masks to isolate actor-shaped regions and bounding boxes."""
        motion_mask = cv.dilate(motion_mask, self._edge_dilation_kernel, iterations=1)
        initial_contours, _ = cv.findContours(
            motion_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        raw_rects = [
            cv.boundingRect(cnt)
            for cnt in initial_contours
            if cv.contourArea(cnt) >= min_actor_area
        ]
        if not raw_rects:
            return np.zeros_like(raw_hsv_mask), []

        spatial_roi = np.zeros_like(cleaned_hsv_mask)
        for x, y, w, h in raw_rects:
            spatial_roi[y : y + h, x : x + w] = 255

        actor_cleaned_hsv = cv.bitwise_and(cleaned_hsv_mask, spatial_roi)
        actor_contours, _ = cv.findContours(
            actor_cleaned_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        solid_actor_mask = np.zeros_like(cleaned_hsv_mask)
        cv.drawContours(
            solid_actor_mask, actor_contours, -1, color=255, thickness=cv.FILLED
        )
        final_actor_mask = cv.bitwise_and(raw_hsv_mask, solid_actor_mask)
        return final_actor_mask, raw_rects

    def _get_candidate_mask(
        self, frame: Frame, region: cv.typing.Rect
    ) -> cv.typing.MatLike:
        """Create a clean candidate mask for a region by combining motion and HSV cues."""
        x, y, w, h = region
        img = frame.image[y : y + h, x : x + w]

        # MOG motion mask
        motion_mask = self._mog.apply(frame.image)
        fg_mask = motion_mask[y : y + h, x : x + w]
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        _, fg_mask = cv.threshold(fg_mask, 125, 255, cv.THRESH_BINARY)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv.dilate(fg_mask, kernel=kernel, iterations=1)

        # HSV masking
        hsv_mask_raw = self._get_hsv_mask(img=img)
        hsv_mask_cleaned = cv.morphologyEx(hsv_mask_raw, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        hsv_mask_cleaned = cv.dilate(hsv_mask_cleaned, kernel, iterations=1)
        hsv_mask_cleaned = cv.morphologyEx(
            hsv_mask_cleaned, cv.MORPH_CLOSE, kernel, iterations=1
        )

        # Combine for best estimate
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
        self, frame: Frame, rect: cv.typing.Rect
    ) -> list[DetectionCandidate]:
        """Create final actor detection candidates by filtering contour geometry."""
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
            if (
                area > self._scaled_min_sprite_area
                and (
                    _MIN_SPRITE_ASPECT_RATIO <= aspect_ratio <= _MAX_SPRITE_ASPECT_RATIO
                )
                and area < self._scaled_max_sprite_area
                and density > _MIN_SPRITE_DENSITY
            ):
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
                    DetectionCandidate(rect=[x, y, w, h], contour=cnt, binary_mask=mask)
                )

        return candidates

    def detect(self, frame: Frame) -> list[DetectionCandidate]:
        """Locate dynamic actor regions for one frame."""

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
            rect = self._rescale_rect(candidate.rect)
            detections.append(
                DetectionCandidate(
                    rect=rect,
                    contour=self._rescale_contour(candidate.contour),
                    binary_mask=self._rescale_mask(
                        candidate.binary_mask, rect[2], rect[3]
                    ),
                )
            )

        return detections
