import cv2
import numpy as np

_hsv_mask_lower_1 = np.array([83,56, 40])
_hsv_mask_upper_1 = np.array([180, 255, 255])
_hsv_mask_lower_2 = np.array([0, 64, 45])
_hsv_mask_upper_2 = np.array([45, 255, 255])
_line_erase_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
def _get_hsv_mask(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        hsv_mask = cv2.inRange(img_temp, _hsv_mask_lower_1, _hsv_mask_upper_1)
        hsv_mask = hsv_mask | cv2.inRange(img_temp, _hsv_mask_lower_2, _hsv_mask_upper_2)

        # Erase thin lines left behind after hsv masking
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, _line_erase_kernel)
        
        # TODO: Erase long horizontal lines with another kernel

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=1)
        return hsv_mask
    
# Open the video file or game capture stream
cap = cv2.VideoCapture(".\\data\\recordings\\great_bay.mp4")

ret, first_frame = cap.read()
if not ret:
    print("Failed to open video stream.")
    exit()

# Resize for faster performance (mimicking Camus' emphasis on efficiency)
scale_width = 640
scale_height = 360
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.resize(prev_gray, (scale_width, scale_height))
prev_frame = cv2.resize(first_frame, (scale_width, scale_height))
# Create an HSV image for flow visualization
hsv = np.zeros_like(first_frame)
hsv = cv2.resize(hsv, (scale_width, scale_height))
hsv[..., 1] = 255  # Set saturation to maximum
fgbg = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=22, detectShadows=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Pre-process current frame
    frame_resized = cv2.resize(frame, (scale_width, scale_height))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    fg_mask = fgbg.apply(frame_resized)
    hsv_thresh = _get_hsv_mask(frame_resized)
    cv2.imshow("HSV threshold", hsv_thresh)

    # Compute dense optical flow (Using DISFlow for ultra-fast, real-time performance)
    # This serves as a modern software equivalent to Camus' real-time constraints
    inst = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_FAST)
    flow = inst.calc(prev_gray, gray, None)

    # Separate the flow into horizontal (u) and vertical (v) components
    u = flow[..., 0]
    v = flow[..., 1]

    # Calculate magnitude and angle of the motion vectors
    magnitude, angle = cv2.cartToPolar(u, v)

    # Camus Application: Threshold low magnitudes to eliminate background/UI noise
    # Adjust '2.0' based on how fast the characters move relative to the stage
    motion_mask = magnitude > 7
    
    color_diff = cv2.absdiff(frame_resized, prev_frame)
    color_diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)

    # Multiply the optical flow magnitude by the color difference
    # High motion + high color contrast = character
    _, character_likelihood = cv2.threshold(color_diff_gray, 20, 255, cv2.THRESH_BINARY)
    character_likelihood = fg_mask
    cv2.imshow("Diff", character_likelihood)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    character_likelihood = cv2.morphologyEx(character_likelihood, cv2.MORPH_OPEN, kernel, iterations=3)
    char_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    character_likelihood = cv2.dilate(character_likelihood, kernel, iterations=2)
    character_likelihood = cv2.morphologyEx(character_likelihood, cv2.MORPH_CLOSE, char_kernel, iterations=1)

    final = character_likelihood & hsv_thresh
    
    final_rgb = cv2.bitwise_and(frame_resized, frame_resized, mask=final)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    # character_likelihood = cv2.morphologyEx(character_likelihood, cv2.MORPH_OPEN, kernel)
    cv2.imshow("character likelihood", final_rgb)

    # Map the optical flow to the HSV color space
    # Hue represents direction, Value (brightness) represents magnitude
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Filter out static areas using our threshold mask
    hsv[..., 2] = np.where(motion_mask, hsv[..., 2], 0)
    
    # Convert HSV to BGR format to display
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # # Show the results
    # cv2.imshow("Original Frame with Motion Vectors", frame_resized)
    cv2.imshow("Optical Flow Magnitude/Direction (HSV)", flow_rgb)

    # Update the previous frame
    prev_gray = gray
    prev_frame = frame_resized
    # Press 'q' to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()