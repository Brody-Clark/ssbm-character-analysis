import cv2 as cv
import numpy as np
from ssbmv.domain.models import (
    Frame,
    Dimension2D,
    Region,
    GameState,
    DetectionCandidate,
    HUDState,
)
import heapq
import logging
from functools import lru_cache
from dataclasses import dataclass, field
from ssbmv.pipeline.stage_hsv_filters import STAGE_HSV_FITLERS

_logger = logging.getLogger(__name__)

_MIN_SPRITE_AREA_RATIO = 0.006


@dataclass(slots=True)
class Candidate:
    rect: cv.typing.Rect = field(default_factory=lambda: [-1, -1, -1, -1])
    mask: cv.typing.MatLike = field(default_factory=list)


class Detector:
    def __init__(self, stage_name: str):
        self._running_background: cv.typing.MatLike = None
        self._max_iterations = 4
        self._min_roi_area_ratio = 5
        self._motion_sub_learn_rate = 0.5
        self._hsv_mask_lower_1 = np.array([83, 56, 40])
        self._hsv_mask_upper_1 = np.array([180, 255, 255])
        self._hsv_mask_lower_2 = np.array([0, 64, 45])
        self._hsv_mask_upper_2 = np.array([45, 255, 255])
        self._hsv_mask_lower_fd = np.array([0, 33, 76])
        self._hsv_mask_upper_fd = np.array([177, 251, 255])
        self._hsv_mask_lower_tmpl = np.array([0, 0, 0])
        self._hsv_mask_upper_tmpl = np.array([153, 126, 255])
        self._vertical_closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
        self._horizontal_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        self._square_kernel = np.ones((8, 8), np.uint8)
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._line_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        self._small_square_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self._mog = cv.createBackgroundSubtractorMOG2(
            history=200, varThreshold=20, detectShadows=False
        )
        self.max_allowed_diff = 25
        self.min_img = None
        self.max_img = None
        self.frames_processed = 0
        self._static_mask = None
        self._prev_frame = None
        self._hsv_filters = STAGE_HSV_FITLERS.get(stage_name, None)
        if self._hsv_filters is None:
            raise (f"Unsupported stage: {stage_name}")

    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, w = img_temp.shape[:2]

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for item in self._hsv_filters:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)

            hsv_mask = cv.inRange(img_temp, lower, upper)
            combined_mask = cv.bitwise_or(combined_mask, hsv_mask)

        # Invert to get remaining elements in scene
        combined_mask = ~combined_mask

        cv.imshow("raw hsv mask", combined_mask)
        return combined_mask

    def _update_static_mask(self, frame_gray: cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Updates pixel ranges and returns static UI mask (255 = Static UI, 0 = Gameplay).
        """
        # 1. Guarantee frame_gray is strictly a 2D array (H, W)
        if frame_gray.ndim == 3:
            if frame_gray.shape[2] == 1:
                frame_gray = frame_gray.squeeze(axis=2)
            else:
                frame_gray = cv.cvtColor(frame_gray, cv.COLOR_BGR2GRAY)

        # 2. Lazy initialization to guarantee exact shape matching on Frame 1
        if self.min_img is None or self.min_img.shape != frame_gray.shape:
            self.min_img = frame_gray.copy()
            self.max_img = frame_gray.copy()

        self.frames_processed += 1

        # 3. Update minimum and maximum seen values at each pixel coordinate
        np.minimum(self.min_img, frame_gray, out=self.min_img)
        np.maximum(self.max_img, frame_gray, out=self.max_img)

        # 4. Compute max variation per pixel across all observed frames
        range_img = cv.subtract(self.max_img, self.min_img)

        # 5. Static pixels: variation <= max_allowed_diff (X)
        #    Dynamic pixels: variation > max_allowed_diff (X)
        _, static_ui_mask = cv.threshold(
            range_img, self.max_allowed_diff, 255, cv.THRESH_BINARY_INV
        )
        return ~static_ui_mask

    def _get_edges(self, img_gray: cv.typing.MatLike) -> cv.typing.MatLike:
        blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv.Canny(blurred, 2, 46)
        # edges = cv.dilate(edges, self._edge_dilation_kernel, iterations=1)
        return edges

    def _get_closed_edges(self, edges: cv.typing.MatLike) -> cv.typing.MatLike:
        dilated = cv.dilate(edges, self._vertical_closing_kernel, iterations=1)
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, self._square_kernel)
        return closed

    def _get_candidates_full_screen(self, frame: cv.typing.MatLike):
        frame_resized = asdfadsf  # TODO
        prev_frame_resized = self._prev_frame
        diff = cv.absdiff(frame_resized, prev_frame_resized)
        diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

        _, character_likelihood = cv.threshold(diff, 20, 255, cv.THRESH_BINARY)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        character_likelihood = cv.morphologyEx(
            character_likelihood, cv.MORPH_OPEN, kernel, iterations=3
        )
        char_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        character_likelihood = cv.dilate(character_likelihood, kernel, iterations=2)
        character_likelihood = cv.morphologyEx(
            character_likelihood, cv.MORPH_CLOSE, char_kernel, iterations=1
        )

        hsv_thresh = self._get_hsv_mask(img=frame_resized)
        final = character_likelihood & hsv_thresh

        contours, _ = cv.findContours(final, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        regions_of_interest = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            if area > 6000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):
                if area < 120000:
                    regions_of_interest.append([x, y, w, h])
                else:
                    continue

        return regions_of_interest

    def _get_regions_of_interest(
        self, img: cv.typing.MatLike, motion_mask: cv.typing.MatLike
    ) -> list[cv.typing.Rect]:
        hsv_mask = self._get_hsv_mask(img=img)
        edges = cv.Canny(hsv_mask, 12, 100)
        edges = cv.dilate(edges, self._edge_dilation_kernel, iterations=1)
        combined = edges & hsv_mask & motion_mask
        closed = self._get_closed_edges(combined)

        contours, hierarchy = cv.findContours(
            img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        regions_of_interest = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            if area > 10000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):
                if area < 120000:
                    regions_of_interest.append([x, y, w, h])
                else:
                    continue

        return regions_of_interest

    @lru_cache(maxsize=8)
    def _get_min_area(self, dim: Dimension2D) -> int:
        return int(_MIN_SPRITE_AREA_RATIO * dim.w * dim.h)

    def extract_clean_actors(
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
        self, frame: Frame, region: cv.typing.Rect
    ) -> cv.typing.MatLike:
        x, y, w, h = region
        img = frame.image[y : y + h, x : x + w]
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg_mask = self._mog.apply(frame.image)
        _, fg_mask = cv.threshold(fg_mask, 125, 255, cv.THRESH_BINARY)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv.dilate(fg_mask, kernel=kernel)
        cv.imshow("FG MASK", fg_mask)

        hsv_mask_raw = self._get_hsv_mask(img=img)
        hsv_mask_cleaned = cv.morphologyEx(hsv_mask_raw, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        hsv_mask_cleaned = cv.dilate(hsv_mask_cleaned, kernel, iterations=1)
        hsv_mask_cleaned = cv.morphologyEx(
            hsv_mask_cleaned, cv.MORPH_CLOSE, kernel, iterations=1
        )

        actor_candidate_mask = hsv_mask_cleaned & fg_mask
        cv.imshow("Actor candidates", actor_candidate_mask)
        final_actor_mask, rects = self.extract_clean_actors(
            actor_candidate_mask, hsv_mask_cleaned, hsv_mask_raw, min_actor_area=2000
        )
        cv.imshow("final actor mask raw", final_actor_mask)

        final_actor_mask = cv.dilate(final_actor_mask, kernel, iterations=1)
        final_actor_mask = cv.morphologyEx(
            final_actor_mask, cv.MORPH_CLOSE, kernel, iterations=1
        )
        return final_actor_mask

    def _get_candidates(
        self, frame: Frame, rect: cv.typing.Rect, debug: bool = False
    ) -> list[DetectionCandidate]:

        candidate_mask = self._get_candidate_mask(frame=frame, region=rect)
        if debug:
            cv.imshow("Detection Candidate Mask", candidate_mask)

        contours, _ = cv.findContours(
            candidate_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        candidates: list[DetectionCandidate] = []
        mask = np.zeros_like(frame.image)
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            bbox_area = w * h
            density = area / bbox_area
            aspect_ratio = float(w) / h
            if area > 2000 and (aspect_ratio <= 1.65 and aspect_ratio >= 0.35):
                if area < 70000 and density > 0.3:
                    cv.drawContours(mask, [cnt], -1, color=(255), thickness=cv.FILLED)
                    r = [x, y, w, h]
                    candidates.append(
                        DetectionCandidate(
                            rect=r,
                            contour=cnt,
                            binary_mask=mask[y : y + h, x : x + w],
                        )
                    )

                else:
                    # TODO: watershed
                    continue

        if debug:
            cv.imshow("Filtered mask", mask)
        return candidates

    def _get_character_HUDs(self, frame, motion_mask) -> list[HUDState]:
        hud_area = frame[536:586, 230:1080]
        hud_mask = ~motion_mask[536:586, 230:1080]
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        hud_mask = cv.morphologyEx(hud_mask, cv.MORPH_OPEN, kernel, iterations=1)
        hud_mask = cv.morphologyEx(hud_mask, cv.MORPH_CLOSE, kernel, iterations=1)
        hud_mask = cv.dilate(hud_mask, kernel=kernel, iterations=8)

        lower1 = np.array([0, 44, 55])
        upper1 = np.array([179, 255, 255])
        hud_hsv = cv.cvtColor(hud_area, cv.COLOR_BGR2HSV)
        hsv_mask = cv.inRange(hud_hsv, lower1, upper1)
        hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel, iterations=1)

        edges = cv.cvtColor(hud_area, cv.COLOR_BGR2GRAY)
        edges = cv.GaussianBlur(edges, (5, 5), 0)

        # Detect and close edges for contour filling
        edges = cv.Canny(edges, threshold1=120, threshold2=265)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))

        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
        # Fill contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros(hud_mask.shape[:2], dtype=np.uint8)
        cv.drawContours(edge_mask, contours, contourIdx=-1, color=255, thickness=-1)

        # Apply motion mask + edge/contour mask + hsv mask to rgb image
        hud_mask = hud_mask & edge_mask & hsv_mask
        hud_area = cv.bitwise_and(hud_area, hud_area, mask=hud_mask)

        # Slice out each UI profile sprite
        candidate_huds = []
        x, y, w, h = 0, 0, 50, 50
        for _ in range(4):
            _, bin_slice = cv.threshold(
                hud_area[y : y + h, x : x + w], 20, 255, cv.THRESH_BINARY
            )
            hud_state = HUDState(0, None, hud_rect=[x, y, w, h])
            candidate_huds.append(hud_state)
            x += 212
        return candidate_huds

    def detect(
        self, frame: Frame, game_state: GameState
    ) -> tuple[list[DetectionCandidate], list[HUDState]]:
        # if not game_state.huds_found:
        #     fg_mask = self._mog.apply(frame.image)
        #     huds = self._get_character_HUDs(frame=frame, motion_mask=fg_mask)
        #     game_state.hud_states = huds

        # TODO: If number of identifed actors < identified HUDS, then global, else local
        # for actor in game_state.active_tracks:
        #     centroid = actor.predicted_centroid
        #     x,y,w,h = actor.current_roi
        #     # Predicted region is ~20% larger
        #     predicted_rect = [centroid[0]-1.1*w/2, centroid[1] - 1.1*h/2 ,1.1*w, 1.1*h]
        #     roi = self._local_search(predicted_rect)

        dimensions = frame.dimensions
        return self._get_candidates(
            frame=frame, rect=[0, 0, dimensions.w, dimensions.h], debug=game_state.debug
        )
