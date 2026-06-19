import cv2 as cv
import numpy as np
from ssbmv.domain.models import Frame, Dimension2D
import heapq
import logging
from functools import lru_cache
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

_MIN_SPRITE_AREA_RATIO = 0.006
_MAX_SPRITE_AREA_RATIO = ...    # TODO

@dataclass(slots=True)
class Candidate:
    score:int = 0
    rect: cv.typing.Rect = [0,0,0,0]
    lod: int = 0

class Detector:
    def __init__(self):
        self._running_background: cv.typing.MatLike = None
        self._max_iterations = 4
        self._min_roi_area_ratio = 5
        self._motion_sub_learn_rate = 0.005
        self._hsv_mask_lower_1 = np.array([83,56, 40])
        self._hsv_mask_upper_1 = np.array([180, 255, 255])
        self._hsv_mask_lower_2 = np.array([0, 64, 45])
        self._hsv_mask_upper_2 = np.array([45, 255, 255])
        self._closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
        self._square_kernel = np.ones((8, 8), np.uint8)
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._line_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
       
    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        hsv_mask = cv.inRange(img_temp, self._hsv_mask_lower_1, self._hsv_mask_upper_1)
        hsv_mask = hsv_mask | cv.inRange(img_temp, self._hsv_mask_lower_2, self._hsv_mask_upper_2)

        # Erase thin lines left behind after hsv masking
        hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, self._line_erase_kernel)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        hsv_mask = cv.dilate(hsv_mask, kernel, iterations=1)
        return hsv_mask
    
    @lru_cache(maxsize=4)
    def _update_static_mask(self, frame_gray: cv.typing.MatLike):
        if self._running_background is None:
            self._running_background = frame_gray.copy().astype(np.float32)
            return np.zeros_like(frame_gray)
        
        # Blend the current frame into long-term memory
        cv.accumulateWeighted(frame_gray, self._running_background, self._motion_sub_learn_rate)
        
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
        dilated = cv.dilate(edges, self._closing_kernel, iterations=1)
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, self._square_kernel)
        return closed
    
    def _get_regions_of_interest(self, img: cv.typing.MatLike, motion_mask: cv.typing.MatLike, lvl_of_detail: int = 1) -> list[cv.typing.Rect]:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hsv_mask = self._get_hsv_mask(img=img)
        # edges = self._get_edges(img_gray=img_gray)
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
                    regions_of_interest.append([x,y,w,h])
                else:
                    continue
        
        return regions_of_interest
    
    def _score(self, frame: Frame, region: cv.typing.Rect, predicted_rois: list[cv.typing.Rect] | None) -> int:
        # Score = w_1 * S_
        pass
    
    @lru_cache(maxsize=8)
    def _get_min_area(self, dim: Dimension2D) -> int:
        return int(_MIN_SPRITE_AREA_RATIO * dim.w * dim.h)
    
    def _get_motion_mask(self, frame: Frame, region: cv.typing.Rect) -> cv.typing.MatLike:
        
        return self._update_static_mask(frame.image)[] # TODO: apply region
    
    def _get_character_mask(self, frame: Frame, region: cv.typing.Rect) -> cv.typing.MatLike:
        img = frame.image[]  # TODO: apply region
        hsv_mask = self._get_hsv_mask(img=img)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = self._get_edges(img_gray=img_gray)
        motion_mask = self._get_motion_mask(frame=frame, region=region)
        combined = edges & hsv_mask & motion_mask
        combined = self._get_closed_edges(combined)
        
    def _get_candidates(self, frame: Frame, region: cv.typing.Rec, predicted_rois: list[cv.typing.Rect] | None = [], lvl_of_detail=1) -> list[Candidate]:
        
        character_mask = self._get_character_mask(frame=frame, region=region)
        contours, hierarchy = cv.findContours(
            character_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        
        candidates: list[Candidate] = []
        for cnt in contours:
            area = cv.contourArea(cnt)    
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            if area > 10000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):  
                if area < 120000:
                    r = [x,y,w,h]
                    score = self._score(frame, region=r, predicted_rois=predicted_rois, aspect_ratio=aspect_ratio, extent=extent) # TODO: implement handling
                    candidates.append(Candidate(rect=r, lod=lvl_of_detail, score=score))
                else:
                    # TODO: break up and add more contours?
                    continue
    
        return candidates
    
    def detect(self, frame: Frame, predicted_rois: list[cv.typing.Rect] | None) -> list[cv.typing.Rect]:
        
        candidates: list[Candidate] = self._get_candidates(frame=frame, predicted_rois=predicted_rois)
        heapq.heapify(candidates)
        
        best = []
        iterations = 0
        min_area = self._get_min_area(frame.dimension)
        while candidates and iterations < self._max_iterations:
            candidate: Candidate = heapq.heappop(candidates)
            if self._meets_criteria(candidate):
                best.append(candidate.rect)
                continue
            if candidate.area / (frame.dimensions.w * frame.dimensions.h) < min_area:
                continue
            last_lod = candidate.lod
            sub_rois = self._get_regions_of_interest(img=candidate, lvl_of_detail=last_lod+1)
            for r in sub_rois:
                score = self._score(frame=frame, region=r)
                heapq.heappush(candidates, Candidate(rect=r, score=score, lod=last_lod+1))
            iterations += 1
        
        return best