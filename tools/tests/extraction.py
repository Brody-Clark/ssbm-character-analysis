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
frame_file = str(frame_dir / 'Region_1.jpg')
frame_rgb = cv.imread(frame_file)
print(frame_rgb.shape)
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