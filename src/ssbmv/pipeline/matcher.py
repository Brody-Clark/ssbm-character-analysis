import logging
import itertools
import numpy as np
import cv2 as cv
from ssbmv.domain.sprite_database import SpriteDatabase, Character
from ssbmv.domain.models import TrackedObjectState, Frame
from skimage.feature import local_binary_pattern

_logger = logging.getLogger(__name__)
    
class Matcher:
    def __init__(self,  sprite_database: SpriteDatabase):
        self._templates: dict[Character, list[list[np.float32]]]  = self._load_templates(sprite_database)
        
    def _get_features(self, image: cv.typing.MatLike):
        contour = self._get_shape_contour(image=image)
        distances = self._compute_centroid_distances(contour=contour)
        desc = self._get_fourier_descriptors(distances=distances, num_descriptors=32, sample_size=128)
        hist = self._extract_character_lbp(image) # TODO: finish
        features = itertools.chain(hist, desc) # TODO: normalize
        return features
    
    def _extract_character_lbp(self, roi_masked_bgr: cv.typing.MatLike, P:int=8, R:int=1):
        # 1. Convert your masked ROI to grayscale
        gray = cv.cvtColor(roi_masked_bgr, cv.COLOR_BGR2GRAY)
        
        # 2. Create a boolean mask of where the character actually is (non-black pixels)
        character_mask = gray > 0
        
        # 3. Compute LBP on the entire grayscale patch
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        
        # 4. CRITICAL: Extract ONLY the LBP values that fall inside the character mask
        # This flattens the array automatically, ignoring all background pixels
        character_lbp_values = lbp[character_mask]
        
        # 5. Calculate the histogram on just the character's texture values
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(character_lbp_values, bins=n_bins, range=(0, n_bins))
        
        # 6. Normalize the histogram so it sums to 1.0
        hist = hist.astype("float32")
        hist /= (hist.sum() + 1e-7)
        
        return hist

    def _load_templates(self, sprite_database: SpriteDatabase) -> dict[Character, list[list[np.float32]]]:
        templates = {}
        for char, sprite_sheet in sprite_database.character_sprite_db.items():
            features = []
            for s in sprite_sheet.sprite_img:
                features.append(self._get_features(image=s))
            templates[char] = features
        return templates

    def _get_shape_contour(self, image: cv.typing.MatLike):
        # Ensure image is binary
        _, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Select the largest contour assuming it's the sprite
        contour = max(contours, key=cv.contourArea)
        return contour[:, 0, :] # Reshape to (N, 2) for easier coordinate math 

    def _compute_centroid_distances(self, contour: cv.typing.MatLike):
        # Calculate moments to find the centroid
        M = cv.moments(contour.astype(np.float32))
        if M['m00'] == 0:
            return None
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        # Compute Euclidean distances from centroid to all boundary points
        distances = np.sqrt((contour[:, 0] - cx)**2 + (contour[:, 1] - cy)**2)
        return distances
    
    def _get_fourier_descriptors(self, distances, num_descriptors: int = 32, sample_size: int = 128):
        # Resample distances to a fixed size for uniform FFT size
        # This grants sampling/scale invariance
        xp = np.linspace(0, len(distances), len(distances))
        x = np.linspace(0, len(distances), sample_size)
        resampled_distances = np.interp(x, xp, distances)
        
        # Compute FFT
        fft_coeffs = np.fft.fft(resampled_distances)
        
        # Take magnitude to achieve rotation invariance (discards phase shifts)
        fft_lengths = np.abs(fft_coeffs)
        
        # Normalize by the DC component (fft_lengths[0]) to achieve scale invariance.
        # Skip the DC component in the final vector as it's now always 1.0
        if fft_lengths[0] == 0:
            return None
        normalized_fds = fft_lengths[1:num_descriptors+1] / fft_lengths[0]
        
        return normalized_fds
    
    def _get_best_match(roi_features, template_dict, threshold=0.15):
        best_match = "Unknown"
        min_distance = float("inf")
        
        for character_name, template_features in template_dict.items():
            # Calculate Euclidean Distance (L2) between the two texture arrays
            distance = np.linalg.norm(roi_features - template_features)
            
            if distance < min_distance:
                min_distance = distance
                best_match = character_name
                
        # Apply a safety gate to reject bad matches
        if min_distance > threshold:
            return "Unknown/Background Stage Geometry"
            
        return best_match

    def match(self, frame: Frame, tracked_objs: list[TrackedObjectState]) -> list[TrackedObjectState]:
        for obj in tracked_objs:
            features = self._get_features(image=obj.region.masked_rgb_slice)
            sprite_name = self._get_best_match(features, self._templates)
            obj.sprite_name
            
            
            
        pass