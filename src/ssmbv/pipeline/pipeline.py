import cv2 as cv
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from collections.abc import Callable
from src.ssmbv.models import Frame, Detection


class _Detector:
    def __init__(self):
        self.hsv_threshold_primary = [np.array([83,56, 40]), np.array([180, 255, 255])]
        self.hsv_threshold_secondary = [np.array([0, 64, 45]), np.array([45, 255, 255])]
    
    def _get_hsv_mask(self, img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        lower_1 = np.array([83,56, 40])
        upper_1 = np.array([180, 255, 255])
        lower_2 = np.array([0, 64, 45])
        upper_2 = np.array([45, 255, 255])
        
        hsv_mask = cv.inRange(img_temp, lower_1, upper_1)
        hsv_mask = hsv_mask | cv.inRange(img_temp, lower_2, upper_2)

        # Erase some of the thin lines left behind after hsv masking
        line_eraser_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, line_eraser_kernel)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        hsv_mask = cv.dilate(hsv_mask, kernel, iterations=1)
        return hsv_mask
    
    def _get_edges(img: cv.typing.MatLike) -> cv.typing.MatLike:
        img_temp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(img_temp, (5, 5), 0)
        edges = cv.Canny(blurred, 18, 100)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        edges = cv.dilate(edges, kernel, iterations=1)
        return edges
    
    def _get_closed_edges(edges: cv.typing.MatLike) -> cv.typing.MatLike:
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
        dilated = cv.dilate(edges, vertical_kernel, iterations=1)

        kernel = np.ones((8, 8), np.uint8)
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
        return closed
    def _get_regions_of_interest(img):
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
            if area > 10000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):  
                if area < 120000:
                    cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                    M = cv.moments(cnt)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.rectangle(result, [x, y, w, h], color=(240, 255, 0), thickness=1)
                    
                    cv.putText(result, f"A: {area:.3f}px", (cx - 30, cy),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    regions_of_interest.append([x,y,w,h])
                else:
                    # break up area vertically???
                    continue
        return regions_of_interest
    
    def detect(self, frame: Frame) -> Detection:
        img = frame.image
       
        hsv_mask = self._get_hsv_mask(img=img)
        edges = self._get_edges(img=img)

        comb = edges & hsv_mask # & dynamic_mask

        comb = self._get_closed_edges(comb)

        regions_of_interest = self._get_regions_of_interest(comb)
        result = frame_rgb.copy()
        for cnt in contours:
            area = cv.contourArea(cnt)    
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h
            extent = float(area) / (w * h)
            
            # Skip tiny noise contours and very large contours
            if area > 10000 and (aspect_ratio >= 0.5 and aspect_ratio <= 3.5):  
                if area < 120000:
                    cv.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                    M = cv.moments(cnt)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv.rectangle(result, [x, y, w, h], color=(240, 255, 0), thickness=1)
                    
                    cv.putText(result, f"A: {area:.3f}px", (cx - 30, cy),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    regions_of_interest.append(frame_rgb[y : y + h, x : x + w])
                    hsv_masks.append(hsv_mask[y: y + h, x : x + w])
                else:
                    # break up area vertically???
                    continue
                
        show(result, 'result')
        for idx, r in enumerate(regions_of_interest):
            show(r, f"ROI {idx}")
            cv.imwrite(str(frame_dir / f'Region_{idx}.png'), r)
        
    
class _Tracker:
    def __init__(self):
        pass
    
class _CharacterClassifier:
    def __init__(self):
        pass
    
class _AnimationClassifier:
    def __init__(self):
        pass

@dataclass(slots=True)
class PipelineConfig:
    classify_animations: bool = True
    classify_characters: bool = True
    track_characters: bool = True

class ObjectType(Enum):
    CHARACTER = 1

@dataclass(slots=True)
class TrackedObjectState:
    track_id: int = -1
    rect: cv.typing.Rect = [-1,-1,-1,-1]
    object_type: ObjectType = ObjectType.CHARACTER
    object_name: Optional[str]
    animation_name: Optional[str]
    
    
@dataclass(slots=True)
class GameState:
    timestamp: int
    objects: list[TrackedObjectState]

class VisionPipeline:
    def __init__(self, config: PipelineConfig):
        self.detector: _Detector = _Detector()
        self.tracker: _Tracker = _Tracker()
        self.character_classifier: _CharacterClassifier =_CharacterClassifier()
        self.animation_classifier: _AnimationClassifier = _AnimationClassifier()
        self._subscribers = []
    
    def subscribe(self, callback: Callable[..., None]) -> None:
        if not callable(callback):
            raise TypeError(
                f"Expected a callable, but received {type(callback).__name__}"
            )

        self._subscribers.append(callback)
        
    def process(frame: Frame) -> GameState:
        