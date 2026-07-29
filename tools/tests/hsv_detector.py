import cv2
import numpy as np
from pathlib import Path
# Global variables to store current mouse position
mouse_x, mouse_y = 0, 0

def posterize_lut(image, levels):
    # Calculate the size of each color bucket division
    bucket_size = 256 // levels
    
    # Create a 256-element lookup array mapping original values to the bucket centers
    lut = np.array([min((i // bucket_size) * bucket_size + (bucket_size // 2), 255) 
                    for i in range(256)]).astype(np.uint8)
    
    # Apply the lookup table instantly across all channels
    return cv2.LUT(image, lut)

def mouse_move(event, x, y, flags, param):
    """Callback function that updates coordinates when the mouse moves."""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


# --- 1. Load Image and Convert to HSV ---
# Replace 'stage_screenshot.png' with your file path
cwd = Path.cwd()
image_path =  cwd / "data" /  "test"/ "great_bay_frame.jpg"
img_bgr = cv2.imread(str(image_path))

if img_bgr is None:
    raise Exception(f"Error: Could not open or find the image '{image_path}'.")

img_bgr = posterize_lut(img_bgr, 5)
out_path = cwd / "data" /  "test"/ "great_bay_frame_posterized.jpg"
cv2.imwrite(str(out_path), img_bgr)

# Convert to HSV color space
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# --- 2. Setup OpenCV Window and Callback ---
window_name = "HSV Eyedropper Tool"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_move)

print("Hover your mouse over the image to inspect HSV values.")
print("Press 'q' or 'ESC' to exit.")

# --- 3. Main Display Loop ---
while True:
    # Clone the original image so we draw fresh text every frame
    display_img = img_bgr.copy()

    # Get dimensions to prevent drawing text off-screen
    h, w, _ = display_img.shape

    # Clamp mouse coordinates to image boundaries
    x = max(0, min(mouse_x, w - 1))
    y = max(0, min(mouse_y, h - 1))

    # Read the HSV values at the current pixel coordinate
    h_val = img_hsv[y, x, 0]
    s_val = img_hsv[y, x, 1]
    v_val = img_hsv[y, x, 2]

    # Create the display text string
    text = f"X:{x} Y:{y} | H:{h_val} S:{s_val} V:{v_val}"

    # Determine text placement (shift text up if mouse is near the bottom edge)
    text_y = y - 15 if y > 30 else y + 25
    text_x = x + 15 if x < w - 250 else x - 250

    # Draw a small crosshair at the pixel
    cv2.drawMarker(
        display_img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA
    )

    # Draw a dark background shadow behind the text for readability
    cv2.putText(
        display_img,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    # Draw the green text overlay
    cv2.putText(
        display_img,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    # Render the updated frame
    cv2.imshow(window_name, display_img)

    # Exit keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()