import cv2 as cv
from pathlib import Path

class EdgeThreshold:
    def __init__(self):
        self.min_e = 0
        self.max_e = 1140

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
cv.createTrackbar('min E', 'image', 0, 1140, onMinEChanged)
cv.createTrackbar('max E', 'image', 255, 1140, onMaxEChanged)

while True:
    cv.imshow('image', edges)

    key = cv.waitKey(1)
    if key == ord(' '):
        break
    
    edges = cv.Canny(blurred, et.min_e, et.max_e)        

cv.destroyAllWindows()

if __name__=="__main__":
    pass
