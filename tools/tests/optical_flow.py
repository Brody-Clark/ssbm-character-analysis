import cv2
import numpy as np

# Load your gameplay video
cap = cv2.VideoCapture("gameplay_footage.mp4")

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Failed to open video.")
    cap.release()
    exit()

# Convert to grayscale (Optical Flow requires single-channel images)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Farneback Dense Optical Flow
    # Returns an array 'flow' of shape (height, width, 2) containing (dx, dy) vectors
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, 
        gray, 
        None, 
        pyr_scale=0.5,   # Image scale (< 1) to build pyramids
        levels=3,        # Number of pyramid layers
        winsize=15,      # Averaging window size
        iterations=3,    # Iterations per pyramid level
        poly_n=5,        # Size of pixel neighborhood for polynomial expansion
        poly_sigma=1.1,  # Gaussian standard deviation for smoothing
        flags=0
    )

    # Split the flow into horizontal (dx) and vertical (dy) components
    dx = flow[..., 0]
    dy = flow[..., 1]

    # Calculate the magnitude (speed) of motion for every pixel
    magnitude = np.sqrt(dx**2 + dy**2)

    # Normalize the magnitude map into a 0-255 grayscale range for visualization
    # Fast-moving pixels (close objects) become bright white. 
    # Slow-moving pixels (distant objects) stay dark.
    depth_map = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)

    # Optional: Apply a colormap to make it look like a thermal depth map
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    # Display the original gameplay next to your calculated depth map
    combined_view = np.hstack((frame, depth_colormap))
    cv2.imshow("Gameplay (Left) vs Estimated Depth Map (Right)", combined_view)

    # Update previous frame for the next iteration
    prev_gray = gray

    # Press 'q' to exit early
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()