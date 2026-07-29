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

_logger = logging.getLogger(__name__)

_MIN_SPRITE_AREA_RATIO = 0.006


@dataclass(slots=True)
class Candidate:
    rect: cv.typing.Rect = [0, 0, 0, 0]
    mask: cv.typing.MatLike = field(default_factory=list)


class Detector:
    def __init__(self):
        self._running_background: cv.typing.MatLike = None
        self._max_iterations = 4
        self._min_roi_area_ratio = 5
        self._motion_sub_learn_rate = 0.005
        self._hsv_mask_lower_1 = np.array([83, 56, 40])
        self._hsv_mask_upper_1 = np.array([180, 255, 255])
        self._hsv_mask_lower_2 = np.array([0, 64, 45])
        self._hsv_mask_upper_2 = np.array([45, 255, 255])
        self._vertical_closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
        self._horizontal_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        self._square_kernel = np.ones((8, 8), np.uint8)
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._line_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        self._mog = cv.createBackgroundSubtractorMOG2(
            history=600, varThreshold=22, detectShadows=True
        )

    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # TODO: should not always convert

        hsv_mask = cv.inRange(img_temp, self._hsv_mask_lower_1, self._hsv_mask_upper_1)
        hsv_mask = hsv_mask | cv.inRange(
            img_temp, self._hsv_mask_lower_2, self._hsv_mask_upper_2
        )

        # Erase thin lines left behind after hsv masking
        hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, self._line_erase_kernel)

        # TODO: Erase long horizontal lines with another kernel

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        hsv_mask = cv.dilate(hsv_mask, kernel, iterations=1)
        return hsv_mask

    def _update_static_mask(self, frame_gray: cv.typing.MatLike):
        if self._running_background is None:
            self._running_background = frame_gray.copy().astype(np.float32)
            return np.zeros_like(frame_gray)

        # Blend the current frame into long-term memory
        cv.accumulateWeighted(
            frame_gray, self._running_background, self._motion_sub_learn_rate
        )

        # Convert back to 8-bit to compare
        bg_model = cv.convertScaleAbs(self._running_background)

        # Take absolute difference between current frame and the long-term stable model
        diff = cv.absdiff(frame_gray, bg_model)

        # Threshold the difference to isolate moving elements
        _, motion_mask = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

        return motion_mask

    def _get_edges(self, img_gray: cv.typing.MatLike) -> cv.typing.MatLike:
        blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv.Canny(blurred, 18, 100)
        edges = cv.dilate(edges, self._edge_dilation_kernel, iterations=1)
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
            if area > 10000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):
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

    def _score(
        self,
        frame: Frame,
        region: cv.typing.Rect,
        predicted_rois: list[cv.typing.Rect] | None,
    ) -> int:
        # Score = w_1 * S_
        return 1

    @lru_cache(maxsize=8)
    def _get_min_area(self, dim: Dimension2D) -> int:
        return int(_MIN_SPRITE_AREA_RATIO * dim.w * dim.h)

    def _get_candidate_mask(
        self, frame: Frame, region: cv.typing.Rect
    ) -> cv.typing.MatLike:
        x, y, w, h = region
        img = frame.image[y : y + h, x : x + w]
        hsv_mask = self._get_hsv_mask(img=img)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = self._get_edges(img_gray=img_gray)
        # motion_mask = self._update_static_mask(img)
        combined = edges & hsv_mask  # & motion_mask
        combined = self._get_closed_edges(combined)

        return combined

    def _get_candidates(
        self, frame: Frame, rect: cv.typing.Rect
    ) -> list[DetectionCandidate]:

        candidate_mask = self._get_candidate_mask(frame=frame, region=rect)
        contours, _ = cv.findContours(
            candidate_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        candidates: list[DetectionCandidate] = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            if area > 10000 and (aspect_ratio <= 0.7 and aspect_ratio >= 0.35):
                if area < 120000:
                    r = [x, y, w, h]
                    candidates.append(
                        DetectionCandidate(
                            rect=r,
                            contour=cnt,
                            binary_mask=candidate_mask[y : y + h, x : x + w],
                        )
                    )
                else:
                    # TODO: watershed
                    continue

        return candidates

    # TODO: **NEW** local and global methods
    def _local_search(self, frame, rect):
        x, y, w, h = rect
        local = frame[y : y + h, x : x + w]
        candidates = self._get_candidates(local)
        return candidates

    def _global_search(self, frame):
        for track in matched_tracks:
            x, y, w, h = track
            local = frame[y : y + h, x : x + w]
        frame_copy = frame.copy()
        cv.rectangle(frame_copy, rec=track, color=(0, 0, 0), thickness=-1)
        frame_copy = cv.resize(
            frame_copy, (320, 180)
        )  # TODO: need to better handle scaling than hard coding
        self._get_candidates(frame_copy)
        # TODO: scale back up.
        candidate_boxes = scale_boxes_up(candidate_boxes_scaled, scale_factor=4)

        pass

    def _get_character_HUDs(self, frame, motion_mask):
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
            candidate_huds.append(bin_slice)
            x += 212
        return candidate_huds

    def detect(self, frame: Frame, game_state: GameState) -> list[DetectionCandidate]:
        # Detect any undiscovered HUD elements
        # fg_mask = self._mog.apply(frame)
        # huds_to_find = [i for i,h in enumerate(game_state.hud_states) if h is None]
        # huds = self._get_character_HUDs(frame=frame, motion_mask=fg_mask)

        # for i in huds_to_find:
        #     contours, _ = cv.findContours(
        #         huds[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        #     )
        #     for contour in contours:
        #         if cv.contourArea(contour=contour) > 60:
        #             game_state.hud_states[i].hud_rect = cv.boundingRect(contour)

        # if game_state.expected_player_count <= game_state.active_tracks
        # for actor in game_state.active_tracks:
        #     centroid = actor.predicted_centroid
        #     x,y,w,h = actor.current_roi
        #     # Predicted region is ~20% larger
        #     predicted_rect = [centroid[0]-1.1*w/2, centroid[1] - 1.1*h/2 ,1.1*w, 1.1*h]
        #     roi = self._local_search(predicted_rect)
        dimensions = frame.dimensions
        return self._get_candidates(
            frame=frame, rect=[0, 0, dimensions[0], dimensions[1]]
        )
