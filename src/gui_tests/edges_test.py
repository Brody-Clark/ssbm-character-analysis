import cv2 as cv
import numpy as np

class EdgeThreshold:
    def __init__(self):
        self.min_e = 20
        self.max_e = 60

et = EdgeThreshold()

def onMinEChanged(x):
    et.min_e = x
def onMaxEChanged(x):
    et.max_e = x

f1 = cv.imread('.\\frames\\frame_01.png')
f2 =  cv.imread('.\\out\\frame0072.png')

f1 = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
f2 = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)
blurred1 = cv.GaussianBlur(f1, (5, 5), 0)
blurred2 = cv.GaussianBlur(f2, (5, 5), 0)
edges1 = cv.Canny(blurred1, 20, 60)
edges2 = cv.Canny(blurred2, 20, 60)
motion = cv.absdiff(edges1, edges2)
cv.namedWindow('image')
cv.createTrackbar('min E', 'image', 20, 255, onMinEChanged)
cv.createTrackbar('max E', 'image', 60, 255, onMaxEChanged)

while True:
    cv.imshow('image', edges2)

    key = cv.waitKey(1)
    if key == ord(' '):
        break
    
    edges1 = cv.Canny(blurred1, et.min_e, et.max_e)
    edges2 = cv.Canny(blurred2, et.min_e, et.max_e)
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
