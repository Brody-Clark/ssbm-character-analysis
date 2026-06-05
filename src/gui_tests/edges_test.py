import cv2 as cv
import numpy as np
from pathlib import Path
class EdgeThreshold:
    def __init__(self):
        self.min_e = 20
        self.max_e = 60

et = EdgeThreshold()

def onMinEChanged(x):
    et.min_e = x
def onMaxEChanged(x):
    et.max_e = x

frame_dir = Path.cwd() / 'data' / 'out'
frame_file = str(frame_dir / 'frame0058.png')
frame_rgb = cv.imread(frame_file)
cv.imshow('rgb', frame_rgb)
frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(frame_gray, (5, 5), 0)
edges = cv.Canny(blurred, 20, 60)
cv.namedWindow('image')
cv.createTrackbar('min E', 'image', 20, 255, onMinEChanged)
cv.createTrackbar('max E', 'image', 60, 255, onMaxEChanged)

while True:
    cv.imshow('image', edges)

    key = cv.waitKey(1)
    if key == ord(' '):
        break
    
    edges = cv.Canny(blurred, et.min_e, et.max_e)
    # motion = cv.absdiff(edges1, edges2)
        
    
# Define a 3x3 High Pass Filter Kernel (Laplacian-style edge detection)
# The sum of all elements in an edge-detection HPF kernel should equal 0
# kernel = np.array([[-1, -1, -1],
#                    [-1,  8, -1],
#                    [-1, -1, -1]])

# # Apply convolution 
# hpf_spatial = cv2.filter2D(gray, -1, kernel)
# hsv_image = cv2.cvtColor(hpf_spatial, cv2.COLOR_GRAY2BGR)
# hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)
# cv2.imshow('hpf', hpf_spatial)


cv.destroyAllWindows()
