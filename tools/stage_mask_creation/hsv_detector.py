import sys
import cv2
import numpy as np
from pathlib import Path

# Global state
frame_dir = Path.cwd() / 'data' /'venom'
frame_file = str(frame_dir / 'frame_0156.jpg')
image_path = frame_file
original_img = None
hsv_img = None

# Mouse ROI selection variables
selecting = False
ix, iy = -1, -1
current_rect = None

# List of saved ROI masks and HSV bounds
# Format: {"box": (x, y, w, h), "lower": [h, s, v], "upper": [h, s, v]}
selected_ranges = []


def extract_hsv_bounds(roi_hsv: np.ndarray) -> tuple[list[int], list[int]]:
    """Calculates min and max HSV bounds for a given HSV region."""
    h_min = int(np.min(roi_hsv[:, :, 0]))
    h_max = int(np.max(roi_hsv[:, :, 0]))
    s_min = int(np.min(roi_hsv[:, :, 1]))
    s_max = int(np.max(roi_hsv[:, :, 1]))
    v_min = int(np.min(roi_hsv[:, :, 2]))
    v_max = int(np.max(roi_hsv[:, :, 2]))

    return [h_min, s_min, v_min], [h_max, s_max, v_max]


def mouse_callback(event, x, y, flags, param):
    """Mouse event callback to handle box selection on the image."""
    global selecting, ix, iy, current_rect, hsv_img

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        ix, iy = x, y
        current_rect = (x, y, 0, 0)

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        w = x - ix
        h = y - iy
        current_rect = (ix, iy, w, h)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        w, h = x2 - x1, y2 - y1

        # Only register selections larger than 3x3 pixels
        if w > 3 and h > 3:
            roi_hsv = hsv_img[y1:y2, x1:x2]
            lower, upper = extract_hsv_bounds(roi_hsv)

            selected_ranges.append(
                {"box": (x1, y1, w, h), "lower": lower, "upper": upper}
            )
            print(
                f"[+] Added Region #{len(selected_ranges)}: Lower={lower}, Upper={upper}"
            )

        current_rect = None


def render_display() -> tuple[np.ndarray, np.ndarray]:
    """Generates the marked selector image and the resulting combined mask."""
    display_img = original_img.copy()
    h, w = original_img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Compute cumulative mask across all selected HSV ranges
    for item in selected_ranges:
        lower = np.array(item["lower"], dtype=np.uint8)
        upper = np.array(item["upper"], dtype=np.uint8)

        mask = cv2.inRange(hsv_img, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Draw selection boxes on display image
    for i, item in enumerate(selected_ranges, 1):
        bx, by, bw, bh = item["box"]
        cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(
            display_img,
            f"#{i}",
            (bx, max(15, by - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Draw active dragging rectangle
    if current_rect is not None:
        rx, ry, rw, rh = current_rect
        cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)

    # Create masked preview (Stage Removed -> Black)
    stage_removed = cv2.bitwise_and(
        original_img, original_img, mask=cv2.bitwise_not(combined_mask)
    )

    return display_img, stage_removed


def main():
    global original_img, hsv_img

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(
            f"Error: Could not load image from '{image_path}'. Check file path!"
        )
        sys.exit(1)

    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

    window_name = "Stage ROI Selector (Click & Drag)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nINSTRUCTIONS:")
    print(" 1. Click & Drag boxes over stage geometry/background elements.")
    print(" 2. Press 'u' to undo the last selection.")
    print(" 3. Press SPACEBAR to finish and output HSV bounds to console.")
    print(" 4. Press 'q' or ESC to quit without saving.\n")

    while True:
        display_img, stage_removed = render_display()

        cv2.imshow(window_name, display_img)
        cv2.imshow("Preview: Stage Geometry Removed", stage_removed)

        key = cv2.waitKey(20) & 0xFF

        # SPACEBAR: Finish and print results
        if key == 32:
            break

        # 'u': Undo last selection box
        elif key == ord("u"):
            if selected_ranges:
                removed = selected_ranges.pop()
                print(f"[-] Removed Region #{len(selected_ranges) + 1}")

        # 'q' or ESC: Quit program
        elif key in (ord("q"), 27):
            print("\nExited without printing final config.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # Output Json
    print("FINAL EXTRACTED HSV BOUNDS FOR PIPELINE:\n")
    print("STAGE_HSV_FILTERS = [")
    for i, item in enumerate(selected_ranges, 1):
        print(f'    {{"lower": {item["lower"]}, "upper": {item["upper"]}}},'
        )
    print("]\n")


if __name__ == "__main__":
    main()