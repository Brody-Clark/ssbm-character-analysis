import cv2
import numpy as np

class HSV:
    def __init__(self):
        self.min_h = 0
        self.max_h = 179
        self.min_s = 0
        self.max_s = 255
        self.min_v = 0
        self.max_v = 255

hsv = HSV()

# 1. Define a dummy callback function required by the trackbar
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

# 2. Build a blank window and an initial black canvas
cv2.namedWindow('image')
img = cv2.imread(".\\out\\frame0038.png")
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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

# 3. Initialize the trackbars (Name, Window, Default, Max, Callback)
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
    # 4. Display the live frame
    cv2.imshow('image', result)
    
    # Check for the Escape key to close the window safely
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    lower = np.array([hsv.min_h, hsv.min_s, hsv.min_v])
    upper = np.array([hsv.max_h, hsv.max_s, hsv.max_v])
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        
    # 5. Extract the position integers of each distinct trackbar
    # r = cv2.getTrackbarPos('R', 'image')
    # g = cv2.getTrackbarPos('G', 'image')
    # b = cv2.getTrackbarPos('B', 'image')
    
    
    # 6. Apply values directly onto the canvas (Note: OpenCV relies on BGR formatting)
    # canvas[:] = [b, g, r]

cv2.destroyAllWindows()
