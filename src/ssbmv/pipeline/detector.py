import cv2 as cv
import numpy as np
from ssbmv.domain.models import Frame, Dimension2D, Region
import heapq
import logging
from functools import lru_cache
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

_MIN_SPRITE_AREA_RATIO = 0.006

@dataclass(slots=True)
class Candidate:
    score:int = 0
    rect: cv.typing.Rect = [0,0,0,0]
    mask: cv.typing.MatLike = field(default_factory=list)

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
        self._vertical_closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
        self._horizontal_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        self._square_kernel = np.ones((8, 8), np.uint8)
        self._edge_dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        self._line_erase_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
       
    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        hsv_mask = cv.inRange(img_temp, self._hsv_mask_lower_1, self._hsv_mask_upper_1)
        hsv_mask = hsv_mask | cv.inRange(img_temp, self._hsv_mask_lower_2, self._hsv_mask_upper_2)

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
        dilated = cv.dilate(edges, self._vertical_closing_kernel, iterations=1)
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, self._square_kernel)
        return closed
    
    def _get_regions_of_interest(self, img: cv.typing.MatLike, motion_mask: cv.typing.MatLike) -> list[cv.typing.Rect]:
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
                    regions_of_interest.append([x,y,w,h])
                else:
                    continue
        
        return regions_of_interest
    
    def _score(self, frame: Frame, region: cv.typing.Rect, predicted_rois: list[cv.typing.Rect] | None) -> int:
        # Score = w_1 * S_
        return 1
    
    @lru_cache(maxsize=8)
    def _get_min_area(self, dim: Dimension2D) -> int:
        return int(_MIN_SPRITE_AREA_RATIO * dim.w * dim.h)

    
    def _get_candidate_mask(self, frame: Frame, region: cv.typing.Rect) -> cv.typing.MatLike:
        x, y, w, h = region
        img = frame.image[y: y + h, x: x + w]
        hsv_mask = self._get_hsv_mask(img=img)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = self._get_edges(img_gray=img_gray)
        motion_mask = self._update_static_mask(img)
        combined = edges & hsv_mask & motion_mask
        combined = self._get_closed_edges(combined)
        
        return combined
        
    def _get_candidates(self, frame: Frame, region: cv.typing.Rect, predicted_rois: list[cv.typing.Rect]) -> list[Candidate]:
        
        candidate_mask = self._get_candidate_mask(frame=frame, region=region)
        contours, _ = cv.findContours(
            candidate_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        
        candidates: list[Candidate] = []
        for cnt in contours:
            area = cv.contourArea(cnt)    
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            if area > 10000 and (aspect_ratio <= 0.7 and aspect_ratio >= .35):  
                if area < 120000:
                    r = [x,y,w,h]
                    score = self._score(frame, region=r, predicted_rois=predicted_rois, aspect_ratio=aspect_ratio, extent=extent)
                    candidates.append(Candidate(rect=r, score=score, mask=candidate_mask))
                else:
                    # TODO: Spectral Clustering for near-convex decomposition
                    continue
    
        return candidates
    
    # TODO: **NEW** local and global methods
    def _local_search(frame, matched_tracks):
        for track in matched_tracks:
            x,y,w,h = track
            local = frame[y:y+h, x:x+w]
            candidate = self._get_candidates(local)
        pass
    
    
    def _global_search(frame, matched_tracks):
        for track in matched_tracks:
            x,y,w,h = track
            local = frame[y:y+h, x:x+w]
        frame_copy = frame.copy()
        cv.rectangle(frame_copy, rec=track, color=(0,0,0))
        frame_copy = cv.resize(frame_copy, (320, 180)) # TODO: need to better handle scaling than hard coding
        self._get_candidates(frame_copy)
        #TODO: scale back up.
        candidate_boxes = scale_boxes_up(candidate_boxes_scaled, scale_factor=4)
        
        pass
    
    def _get_character_HUDs(self, frame, motion_mask):
        # TODO: Bottom 1/4 (?) of screen, look for distinct contours
        hud_area = motion_mask[] # TODO: slice
        contours, _ = cv.findContours(
            ~hud_area, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if len(contours) <= 1:
            return None
        
        candidate_HUDs = 0
        for c in contours:
            x,y,w,h = cv.boundingRect(c)
            # TODO: check size, aspect ratio, etc... Maybe even just edge detection with template matching...
            candidate_HUDs+=1
            
        if candidate_HUDs > 1 and candidate_HUDs < 5:
            return candidate_HUDs
        # TODO: either need to find better way to isolate them, or just assume 4 (bad performance)
        return None
        
    def detect(self, frame: Frame, matched_tracks: list[cv.typing.Rect] | None) -> list[Region]:
        motion_mask = self._update_static_mask(frame)
        if not self._identified_character_HUDs:
           huds = self._get_character_HUDs(frame=frame, motion_mask=motion_mask)
           if huds:
               self._character_HUDs = huds
               self._idetified_character_HUDS = True
           else:
               return []
           
        if len(matched_tracks) == self._character_HUDs:
            candidates = self._local_search(matched_tracks)
        else:
            candidates = self._global_search(matched_tracks)
           
        candidates: list[Candidate] = self._get_candidates(frame=frame, predicted_rois=predicted_rois)
  