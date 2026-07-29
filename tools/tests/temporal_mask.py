import cv2
import numpy as np

motion_sub_learn_rate = 0.005
running_mask = None


def update_static_mask(frame_gray: cv2.typing.MatLike):
    global running_mask
    if running_mask is None:
        running_mask = frame_gray.copy().astype(np.uint8)
        return np.zeros_like(frame_gray)

    # Blend the current frame into long-term memory
    # cv2.accumulateWeighted(frame_gray, running_mask, motion_sub_learn_rate)
    running_mask = running_mask | frame_gray

    # Convert back to 8-bit to compare
    bg_model = cv2.convertScaleAbs(running_mask)

    # Take absolute difference between current frame and the long-term stable model
    diff = cv2.absdiff(frame_gray, bg_model)

    # Threshold the difference to isolate moving elements
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    return motion_mask


def get_character_HUDs(frame, motion_mask):
    hud_area = frame[536:586, 230:1080]
    hud_mask = ~motion_mask[536:586, 230:1080]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hud_mask = cv2.morphologyEx(hud_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    hud_mask = cv2.morphologyEx(hud_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    hud_mask = cv2.dilate(hud_mask, kernel=kernel, iterations=8)

    lower1 = np.array([0, 44, 55])
    upper1 = np.array([179, 255, 255])
    hud_hsv = cv2.cvtColor(hud_area, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hud_hsv, lower1, upper1)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    gray = cv2.cvtColor(hud_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect and close edges for contour filling
    edges = cv2.Canny(blurred, threshold1=120, threshold2=265)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("EDGES", edges)
    # Fill contours
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    edge_mask = np.zeros(hud_mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(edge_mask, contours, contourIdx=-1, color=255, thickness=-1)

    # Apply motion mask + edge/contour mask to rgb image
    hud_mask = hud_mask & edge_mask & hsv_mask
    hud_area = cv2.bitwise_and(hud_area, hud_area, mask=hud_mask)
    cv2.imshow("Masked", hud_area)

    x, y, w, h = 0, 0, 50, 50
    for i in range(4):
        _, bin_slice = cv2.threshold(
            hud_area[y : y + h, x : x + w], 20, 255, cv2.THRESH_BINARY
        )
        cv2.imshow(f"HUD {i}", bin_slice)
        x += 212


# Open the video file or game capture stream
cap = cv2.VideoCapture(".\\data\\recordings\\ssbm_corneria.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=800, varThreshold=46, detectShadows=True
)
ret, prev_frame = cap.read()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame_gray, prev_frame_gray)
    _, motion_mask = cv2.threshold(diff, 65, 255, cv2.THRESH_BINARY)
    cv2.imshow("abs diff", motion_mask)
    fg_mask = fgbg.apply(frame)
    _, motion_mask = cv2.threshold(fg_mask, 129, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed_edges = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("MOG foreground mask", motion_mask)
    mask = update_static_mask(frame_gray=frame_gray)
    get_character_HUDs(frame, fg_mask)

    # cv2.imshow("Temporal Mask", mask)
    prev_frame = frame.copy()

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
