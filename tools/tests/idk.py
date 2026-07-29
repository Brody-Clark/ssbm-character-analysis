import cv2
import numpy as np

# Load video or screen capture stream
cap = cv2.VideoCapture(".\\data\\recordings\\great_bay.mp4")

# Parameters
BUFFER_SIZE = 10     # Number of frames to build the temporal trajectory
MAX_FEATURES = 90   # Sparse points to track across the screen
RANK_CONSTRAINT = 4  # Maximum algebraic rank for a rigid background
ERROR_THRESHOLD = 24.0 # Pixel distance error to flag an independent moving object

# Tracking configuration
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize rolling buffers
frame_history = []
points_history = [] # Will hold a list of point arrays for each frame in the buffer

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_history.append(gray)
    
    # Keep our rolling frame window fixed
    if len(frame_history) > BUFFER_SIZE:
        frame_history.pop(0)
        
    # --- Step 1: Generate or Track Features ---
    if len(frame_history) == 1:
        # Initial frame: Find distinct geometric points across the stage
        p0 = cv2.goodFeaturesToTrack(gray, maxCorners=MAX_FEATURES, qualityLevel=0.01, minDistance=10)
        points_history = [p0]
        continue

    # Track points from the previous frame into the new frame
    p_prev = points_history[-1]
    p_next, st, _ = cv2.calcOpticalFlowPyrLK(frame_history[-2], gray, p_prev, None, **lk_params)
    
    # Filter out lost tracking points to keep the historical tracks synchronized
    valid = (st == 1).reshape(-1)
    
    # Re-align all historical tracking frames to match only the points still actively tracked
    for i in range(len(points_history)):
        points_history[i] = points_history[i][valid]
    p_next = p_next[valid]
    points_history.append(p_next)
    
    if len(points_history) > BUFFER_SIZE:
        points_history.pop(0)

    # --- Step 2 & 3: Build Trajectory Matrix & Apply Rank SVD ---
    # We need a full window buffer to calculate matrix rank dependencies
    if len(points_history) == BUFFER_SIZE:
        num_points = points_history[0].shape[0]
        
        if num_points > RANK_CONSTRAINT:
            # Construct Trajectory Matrix W (Size: 2N x P)
            # Row 0 to N-1: X coordinates through time. Row N to 2N-1: Y coordinates.
            W = np.zeros((2 * BUFFER_SIZE, num_points))
            for f in range(BUFFER_SIZE):
                W[f, :] = points_history[f][:, 0, 0]              # X paths
                W[f + BUFFER_SIZE, :] = points_history[f][:, 0, 1] # Y paths
                
            # Subtract the mean of each row to center the coordinate trajectories
            W_centered = W - np.mean(W, axis=1, keepdims=True)
            
            # Compute Singular Value Decomposition
            U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
            
            # Enforce Rank Constraint: Keep only the top 4 structural singular values
            S_clean = np.zeros_like(S)
            S_clean[:RANK_CONSTRAINT] = S[:RANK_CONSTRAINT]
            
            # Reconstruct the "ideal rigid background representation" of the matrix
            W_reconstructed = U @ np.diag(S_clean) @ Vt
            
            # --- Step 4: Outlier Detection ---
            # Calculate the Euclidean distance error between actual and reconstructed tracks
            errors = np.zeros(num_points)
            for f in range(BUFFER_SIZE):
                dx = W_centered[f, :] - W_reconstructed[f, :]
                dy = W_centered[f + BUFFER_SIZE, :] - W_reconstructed[f + BUFFER_SIZE, :]
                errors += np.sqrt(dx**2 + dy**2)
            errors /= BUFFER_SIZE # Average error per frame for each point
            
            # Draw the results on the live display frame
            current_points = points_history[-1]
            for idx in range(num_points):
                x, y = int(current_points[idx, 0, 0]), int(current_points[idx, 0, 1])
                
                if errors[idx] > ERROR_THRESHOLD:
                    # Point violates background camera constraints -> Independent Motion (Character)
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1) # Red Circle
                    cv2.putText(frame, "Fighter", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    # Point follows camera constraints smoothly -> Rigid Background (Stage)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1) # Green Circle

        # Frequently regenerate fresh features to replace tracks that exit the screen
        if num_points < (MAX_FEATURES * 0.6):
            p_new = cv2.goodFeaturesToTrack(gray, maxCorners=int(MAX_FEATURES - num_points), qualityLevel=0.01, minDistance=10)
            if p_new is not None:
                # Pad past history buffers with the initialization coordinates to match lengths
                for i in range(len(points_history)):
                    points_history[i] = np.vstack((points_history[i], p_new))

    cv2.imshow("Sheikh & Kanade Subspace Background Subtraction", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()