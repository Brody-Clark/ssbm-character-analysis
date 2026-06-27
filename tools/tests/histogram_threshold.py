import cv2
import numpy as np
from pathlib import Path

class HSV:
    def __init__(self):
        self.min_h = 0
        self.max_h = 179
        self.min_s = 0
        self.max_s = 255
        self.min_v = 0
        self.max_v = 255

hsv = HSV()

def onMinHChanged(x):
    hsv.min_h = x
def onMaxHChanged(x):
    hsv.max_h = x
def onMinSChanged(x):
    hsv.min_s = x
def onMaxSChanged(x):
    hsv.max_s = x
def onMinVChanged(x):
    hsv.min_v = x
def onMaxVChanged(x):
    hsv.max_v = x

# cv2.namedWindow('image')
frame_dir = Path.cwd() / 'data' / 'out'
frame_file = str(frame_dir / 'mario_sprite_taunt_01.jpg')
frame_rgb = cv2.imread(frame_file)

hsv_image = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)

frame_file = str(frame_dir / 'Mask_1.jpg')
mask = cv2.imread(frame_file)
# cv2.createTrackbar('min H', 'image', 0, 179, onMinHChanged)
# cv2.createTrackbar('max H', 'image', 179, 179, onMaxHChanged)
# cv2.createTrackbar('min S', 'image', 0, 255, onMinSChanged)
# cv2.createTrackbar('max S', 'image', 255, 255, onMaxSChanged)
# cv2.createTrackbar('min V', 'image', 0, 255, onMinVChanged)
# cv2.createTrackbar('max V', 'image', 255, 255, onMaxVChanged)

sprite_hist_rgb = cv2.calcHist([frame_rgb], [1], None, [32], [0, 256])
cv2.normalize(sprite_hist_rgb, sprite_hist_rgb, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('sprite hist RGB', sprite_hist_rgb)


sprite_hist = cv2.calcHist([hsv_image], [1], None, [32], [0, 256])
cv2.imshow('sprite hsv', hsv_image)
cv2.normalize(sprite_hist, sprite_hist, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('sprite hist', sprite_hist)
# 3. Extract your current cluttered ROI from the video stream
frame_file = str(frame_dir / 'Region_1.jpg')
roi_bgr = cv2.imread(frame_file)
roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

roi_hsv = roi_hsv & mask
cv2.imshow('ROI HSV', roi_hsv)
# 4. Perform the "Intersection Lookup" (Backprojection)
# This calculates the probability of each ROI pixel belonging to the sprite hist
probability_map = cv2.calcBackProject([roi_hsv], [0, 1], sprite_hist, [0, 180, 0, 256], 1)
cv2.imshow('prob map', probability_map)
# 5. Clean up the resulting map with a quick threshold
_, isolated_character_mask = cv2.threshold(probability_map, 80, 255, cv2.THRESH_BINARY)
cv2.imshow('character map isolated', isolated_character_mask)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    
print("Press 'ESC' to exit.")

lower = np.array([hsv.min_h, hsv.min_s, hsv.min_v])
upper = np.array([hsv.max_h, hsv.max_s, hsv.max_v])
mask = cv2.inRange(hsv_image, lower, upper)
result = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

while True:
    cv2.imshow('image', result)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    lower = np.array([hsv.min_h, hsv.min_s, hsv.min_v])
    upper = np.array([hsv.max_h, hsv.max_s, hsv.max_v])
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        
cv2.destroyAllWindows()
