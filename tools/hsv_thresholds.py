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

cv2.namedWindow('image')
frame_dir = Path.cwd() / 'data' / 'out'
frame_file = str(frame_dir / 'frame0058.png')
frame_rgb = cv2.imread(frame_file)

hsv_image = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

cv2.createTrackbar('min H', 'image', 0, 179, onMinHChanged)
cv2.createTrackbar('max H', 'image', 179, 179, onMaxHChanged)
cv2.createTrackbar('min S', 'image', 0, 255, onMinSChanged)
cv2.createTrackbar('max S', 'image', 255, 255, onMaxSChanged)
cv2.createTrackbar('min V', 'image', 0, 255, onMinVChanged)
cv2.createTrackbar('max V', 'image', 255, 255, onMaxVChanged)


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
