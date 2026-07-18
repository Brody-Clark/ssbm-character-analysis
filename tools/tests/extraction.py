import cv2 as cv
import numpy as np
from pathlib import Path

def show(img: any, title: str):
    cv.imshow(title, img)

def wait():
    while True:
        key = cv.waitKey()
        if key == ord(' '):
            return
def clean_watershed_split(roi_bgr, binary_merged_mask):
    """
    Splits merged contours instantly using the distance transform 
    as markers for the Watershed algorithm.
    """
    # 1. Compute Distance Transform
    dist_transform = cv.distanceTransform(binary_merged_mask, cv.DIST_L2, 5)
    
    # 2. Extract the local peak seeds (Sure Foreground)
    # 35% of max distance is usually the sweet spot for distinct centers
    _, sure_fg = cv.threshold(dist_transform, 0.75 * dist_transform.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Find the unknown region (where borders meet)
    # We get this by subtracting the sure foreground from the original mask
    unknown = cv.subtract(binary_merged_mask, sure_fg)
    
    # 4. Label the seeds (each peak gets a unique integer index: 1, 2, 3...)
    num_labels, markers = cv.connectedComponents(sure_fg)
    
    # Add 1 to all labels so that the background is 1 instead of 0
    # (Watershed treats 0 as completely unknown pixels to be classified)
    markers = markers + 1
    
    # Mark the unknown transition boundary regions with 0
    markers[unknown == 255] = 0
    
    # 5. Run Watershed on the BGR image patch using our marker seeds
    # This floods the markers outward until they fill the original mask bounds
    markers = cv.watershed(roi_bgr, markers)
    
    # 6. Extract the separated masks
    # Watershed returns -1 on the boundary lines (dams) it created
    isolated_masks = []
    for label_idx in range(2, num_labels + 1):  # Start at 2 (1 is the background)
        # Create a clean mask for this specific flooded region
        candidate_mask = np.zeros_like(binary_merged_mask)
        candidate_mask[markers == label_idx] = 255
        
        # Quick size filter to discard tiny noise fragments
        if cv.countNonZero(candidate_mask) > 150:
            isolated_masks.append(candidate_mask)
            
    return isolated_masks

def posterize_roi(roi_bgr, factor=64):
    """
    Reduces the color space by flattening pixel shades.
    factor=64 reduces 256 colors down to 4 distinct steps per channel.
    """
    # Integer division drops the low-order bits (shading/gradients)
    quantized = (roi_bgr // factor) * factor
    return quantized
        
frame_dir = Path.cwd() / 'data' / 'out'
frame_file = str(frame_dir / 'Region_1_big.jpg')
frame_rgb = cv.imread(frame_file)
frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
frame_gray = cv.GaussianBlur(frame_gray, (9,9), 0)
frame_hsv = cv.cvtColor(frame_rgb, cv.COLOR_BGR2HSV)
lower1 = np.array([83,56, 40])
upper1 = np.array([180, 255, 255])
lower2 = np.array([0, 64, 45])
upper2 = np.array([45, 255, 255])
hsv_mask = cv.inRange(frame_hsv, lower1, upper1)
hsv_mask = hsv_mask | cv.inRange(frame_hsv, lower2, upper2)
hsv_mask_orig = hsv_mask
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))

new_masks = clean_watershed_split(cv.bitwise_and(frame_rgb, frame_rgb, mask=hsv_mask_orig), hsv_mask_orig)
for i,mask in enumerate(new_masks):
    mask = cv.dilate(mask, kernel, iterations=2)
    cv.imshow(f"mask {i}", mask)
wait()
# Erase some of the thin lines left behind after hsv masking
line_eraser_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, line_eraser_kernel)
hsv_mask = cv.erode(hsv_mask, kernel, iterations=5 )
show(hsv_mask, 'hsv_mask')
hsv_mask = cv.dilate(hsv_mask, kernel, iterations=1)
wait()




flat_color = posterize_roi(frame_rgb)
show(flat_color, "Posterized")
contours, _ = cv.findContours(hsv_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(frame_rgb, contours=contours, contourIdx=-1, color=(223, 220, 20), thickness=cv.FILLED)
show(frame_rgb, "final")




gray = cv.cvtColor(flat_color, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
fm = cv.Laplacian(gray, cv.CV_64F).var()
cv.imshow("laplacina ", fm)
wait()
gray = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(gray, 16, 32)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

show(edges, "edgs")
# 2. Find contours with complete hierarchical relationships
# cv.RETR_CCOMP organizes topology into a 2-level hierarchy (external and internal)
contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

sub_candidates = []
if hierarchy is not None:
    for i, contour in enumerate(contours):
        # Optional: Check if it is an internal child component
        # hierarchy[0][i][3] != -1 means it has a parent (it's inside the main fused box)
        
        area = cv.contourArea(contour)
        if area > 200:  # Filter out tiny pixel fragments
            rec = cv.boundingRect(contour)
            cv.rectangle(frame_rgb, rec=rec, color=(220, 230, 36), thickness=2)
            sub_candidates.append(contour)
            
# cv.drawContours(frame_rgb, sub_candidates, -1, (220, 220, 32), cv.FILLED )

frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
frame_hsv = cv.cvtColor(frame_rgb, cv.COLOR_BGR2HSV)

blurred = cv.GaussianBlur(frame_gray, (5, 5), 0)
edges = cv.Canny(blurred, 12, 64)
show(edges, "edges")
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
dilated_image = cv.dilate(edges, kernel, iterations=1)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 7))
dilated_image = cv.dilate(edges, kernel, iterations=1)
show(dilated_image, "dlaetd")
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))
closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
show(closed, "closed image")

contours, hierarchy = cv.findContours(
    closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

cv.drawContours(frame_rgb, contours=contours, contourIdx=-1, color=(245, 245, 0), thickness=3 )
show(frame_rgb, "final")

for i, c in enumerate(contours):
    x, y, w, h = cv.boundingRect(c)
    show(frame_rgb[y: y + h, x: x + w], f"IMG_{i}")
wait()