import cv2 as cv
import numpy as np
from ssbmv.domain.models import Frame
import heapq
import logging

_logger = logging.getLogger(__name__)


_MAX_ITERATIONS = 8
_MIN_REGION_AREA_RATIO = 5
_HSV_MASK_LOWER_1 = np.array([83,56, 40])
_HSV_MASK_UPPER_1 = np.array([180, 255, 255])
_HSV_MASK_LOWER_2 = np.array([0, 64, 45])
_HSV_MASK_UPPER_2 = np.array([45, 255, 255])
        
class Detector:
    def __init__(self):
       pass
    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        hsv_mask = cv.inRange(img_temp, _HSV_MASK_LOWER_1, _HSV_MASK_UPPER_1)
        hsv_mask = hsv_mask | cv.inRange(img_temp, _HSV_MASK_LOWER_2, _HSV_MASK_UPPER_2)

        # Erase thin lines left behind after hsv masking
        line_eraser_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, line_eraser_kernel)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        hsv_mask = cv.dilate(hsv_mask, kernel, iterations=1)
        return hsv_mask
    
    def _get_edges(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(img_temp, (5, 5), 0)
        edges = cv.Canny(blurred, 18, 100)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        edges = cv.dilate(edges, kernel, iterations=1)
        return edges
    
    def _get_closed_edges(self, edges: cv.typing.MatLike) -> cv.typing.MatLike:
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
        dilated = cv.dilate(edges, vertical_kernel, iterations=1)

        kernel = np.ones((8, 8), np.uint8)
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
        return closed
    
    def _get_regions_of_interest(self, img) -> list[cv.typing.Rect]:
        contours, hierarchy = cv.findContours(
            img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        regions_of_interest = []
        for cnt in contours:
            area = cv.contourArea(cnt)    
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            
            # Skip tiny noise contours and very large contours
            # TODO: replace numbers with image ratios
            if area > 10000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):  
                if area < 120000:
                    # cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                    # M = cv.moments(cnt)
                    # cx = int(M["m10"] / M["m00"])
                    # cy = int(M["m01"] / M["m00"])
                    # cv.rectangle(result, [x, y, w, h], color=(240, 255, 0), thickness=1)
                    
                    # cv.putText(result, f"A: {area:.3f}px", (cx - 30, cy),
                    #             cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    regions_of_interest.append([x,y,w,h])
                else:
                    # break up area vertically???
                    continue
        
        return regions_of_interest
    def _score(self, candidate: cv.typing.Rect) -> int:
        pass
    
    def _get_candidates(self, frame: Frame) -> list[cv.typing.Rect]:
        img = frame.image
        
        hsv_mask = self._get_hsv_mask(img=img)
        edges = self._get_edges(img=img)

        combined = edges & hsv_mask # & dynamic_mask
        combined = self._get_closed_edges(combined)

        return self._get_regions_of_interest(combined)
    
    def detect(self, frame: Frame) -> list[cv.typing.Rect]:
        
        candidates: list[cv.typing.Rect] = self._get_candidates(frame=frame)
        min_area = self._get_min_area(frame.dimension)
        pq = [self._score(c) for c in candidates]
        heapq.heapify(pq) # TODO: need to create custom candidate struct for heap comparison?
        best = []
        iterations = 0
        while pq and iterations < _MAX_ITERATIONS:
            candidate = heapq.heappop(pq)
            if self._meets_criteria(candidate):
                best.append(candidate)
                continue
            if candidate.area / (frame.dimensions.w * frame.dimensions.h) < _MIN_REGION_AREA_RATIO:
                continue
            sub_rois = self._refine(candidate)
            for r in sub_rois:
                process(r)
                heapq.heappush(pq, r)
            iterations += 1
        
        return best