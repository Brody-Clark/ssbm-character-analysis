import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from pathlib import Path
import numpy as np


def extract_character_only_lbp(roi_masked_bgr, P=8, R=1):
    # 1. Convert your masked ROI to grayscale
    gray = cv2.cvtColor(roi_masked_bgr, cv2.COLOR_BGR2GRAY)
    
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

cwd = Path.cwd()
frame = cwd / "data" / "out" / "mario_sprite_idle_01.png"

# 1. Load image and convert to grayscale
image = cv2.imread(str(frame))

# 2. Set LBP parameters
# Radius of the circle and number of neighbors to sample
radius = 1
n_points = 8 * radius

# 3. Compute LBP using the uniform method (rotation-invariant and robust)
hist = extract_character_only_lbp(image, n_points, radius)

cv2.imshow("hist", hist)
print(hist)

while True:
    k = cv2.waitKey(1)
    if k == ord(' '):
        break
    
cv2.destroyAllWindows()