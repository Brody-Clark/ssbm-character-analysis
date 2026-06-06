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
frame_dir = Path.cwd() / 'data' / 'out'
frame_file = str(frame_dir / 'frame0022.png')
frame_rgb = cv.imread(frame_file)
frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
frame_hsv = cv.cvtColor(frame_rgb, cv.COLOR_BGR2HSV)

lower1 = np.array([83,56, 40])
upper1 = np.array([180, 255, 255])
lower2 = np.array([0, 64, 45])
upper2 = np.array([45, 255, 255])
hsv_mask = cv.inRange(frame_hsv, lower1, upper1)
hsv_mask = hsv_mask | cv.inRange(frame_hsv, lower2, upper2)
# show(hsv_mask, 'hsv')

# Get dynamic HSV make
# Attempts to remove stage hues by finding dominant hues in histogram
# -- Has a tendency to remove important character values
# small_hsv = cv.resize(frame_hsv, (160, 90))
# # show(small_hsv, 'small')
# hist = cv.calcHist([small_hsv], [0], None, [180], [0, 180])
# # Get the top 2 most common Hues (likely sky and stage)
# dominant_hues = hist.flatten().argsort()[-3:]
# dynamic_mask = np.ones(frame_hsv.shape[:2], dtype=np.uint8) * 255
# bg_full_mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)

# # You may want to drop a dominant hue if it's 0 (often just pure black borders/artifacts)
# for hue in dominant_hues:
#     # A tolerance of 8-12 usually captures lighting variances on the stage
#     lower_h = np.array([max(0, hue - 10), 40, 40])  # Add low S/V gates to avoid killing white/black game UI
#     upper_h = np.array([min(180, hue + 10), 255, 255])
    
#     bg_color_mask = cv.inRange(frame_hsv, lower_h, upper_h)
    
#     # Combine this background color into our total background map
#     bg_full_mask = cv.bitwise_or(bg_full_mask, bg_color_mask)

# # 4. Invert it ONCE: Keep only what IS NOT background
# dynamic_mask = cv.bitwise_not(bg_full_mask)

# show(dynamic_mask, "dynamic hsv mask")

# Erase some of the thin lines left behind after hsv masking
line_eraser_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
hsv_mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, line_eraser_kernel)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
hsv_mask = cv.dilate(hsv_mask, kernel, iterations=1)

show(hsv_mask, 'hsv_mask')

blurred = cv.GaussianBlur(frame_gray, (5, 5), 0)
edges = cv.Canny(blurred, 18, 100)
# show(edges, "raw edges")

# Dilate edges to increase thickness and give more area for AND operation
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
dilated_image = cv.dilate(edges, kernel, iterations=1)
# show(dilated_image, "dilated edges")

comb = dilated_image & hsv_mask # & dynamic_mask
show(comb, 'combined')

# Dilate and close image to give generous enough bounding boxes
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
comb = cv.dilate(comb, vertical_kernel, iterations=1)
show(comb, 'dilate verticla')

kernel = np.ones((8, 8), np.uint8)
closed_image = cv.morphologyEx(comb, cv.MORPH_CLOSE, kernel)
show(closed_image, "closed image")

# Find contours
contours, hierarchy = cv.findContours(
    closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

regions_of_interest = []
hsv_masks = []
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
wait()

# WIP
# Take a know ROI and try to extract a character
mario_rgb = regions_of_interest[1]
mario_gray = cv.cvtColor(mario_rgb, cv.COLOR_BGR2GRAY)

# _, roi_binary = cv.threshold(mario_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# show(roi_binary, 'bianry')
# line_eraser_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# out = cv.erode(roi_binary, line_eraser_kernel, 1)    
# show(out,'eroded')
# clean_roi = cv.morphologyEx(roi_binary, cv.MORPH_OPEN, line_eraser_kernel)
# show(clean_roi, 'clean')

regions_of_interest[1] = cv.GaussianBlur(regions_of_interest[1],(11, 11), 0 )
mario_edges = cv.Canny(regions_of_interest[1], 0, 48)
# show(mario_edges, "edges before")
lines = cv.HoughLinesP(mario_edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=30, maxLineGap=5)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw a black line to erase the background vector from the edge map
        # A thickness of 3-5 ensures the edge and its immediate dilation neighbors are killed
        cv.line(mario_edges, (x1, y1), (x2, y2), 0, thickness=4)
# show(mario_edges,  'edges AFTER')

vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 7))
mario_edges = cv.dilate(mario_edges, vertical_kernel, iterations=1)
# mario_edges = cv.morphologyEx(mario_edges, cv.MORPH_OPEN, line_eraser_kernel)
# mario_edges = mario_edges & roi_binary
# mario_edges = cv.dilate(mario_edges, vertical_kernel, iterations=1)
mario_edges = mario_edges & hsv_masks[1]
# show(mario_edges, 'Dilaetd edges')
# mario_edges = cv.morphologyEx(mario_edges, cv.MORPH_CLOSE, vertical_kernel)
contours, hierarchy = cv.findContours(
    mario_edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
)


for cnt in contours:
    area = cv.contourArea(cnt)    
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h)
    if area > 8000:  # Skip tiny noise contours and very large contours
        if area < 120000:
            cv.drawContours(mario_rgb, [cnt], -1, (0, 255, 0), 2)
            M = cv.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.rectangle(mario_rgb, [x, y, w, h], color=(240, 255, 0), thickness=1)
            
            cv.putText(mario_rgb, f"A: {area:.3f}px", (cx - 30, cy),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
        else:
            # break up area vertically???
            continue



# cv.imshow('result', mario_rgb)

wait()