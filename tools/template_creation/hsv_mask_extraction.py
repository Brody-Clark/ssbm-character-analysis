"""
Prompts user to specify regions to create HSV masks from and masks remaining images for template creation.
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create hsv masks from an image for template creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input data folder",
        required=True,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output folder",
        required=True,
    )

    args = vars(parser.parse_args())

    input_dir = Path(args["input"]).resolve()

    output_dir = Path(args["output"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pts = []
    current_polygons = []

    def draw_polygon(event, x, y, flags, param):
        global pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            len(pts) > 0 and pts.pop()

    img_paths = [f for f in input_dir.rglob("*") if f.is_file()]
    if not img_paths:
        print("No images found!")
        exit()

    first_img = cv2.imread(img_paths[0])
    hsv_first = cv2.cvtColor(first_img, cv2.COLOR_BGR2HSV)
    clone = first_img.copy()

    window_name = "Step 1: Draw Regions for Histogram Training"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    # This list will hold the normalized color histograms for each feature
    trained_histograms = []

    print("=== INSTRUCTIONS ===")
    print("1. Outline a specific feature (e.g., just the Red shirt).")
    print("2. Press 'n' to calculate its HSV histogram profile and clear canvas.")
    print("3. Repeat for other features (Overalls, Gloves, Face, etc.).")
    print("4. Press 'm' to automatically backproject and extract from all frames.")
    print("====================\n")

    while True:
        img_display = clone.copy()
        for poly in current_polygons:
            cv2.polylines(img_display, [np.array(poly)], True, (255, 0, 0), 1)
        if len(pts) > 0:
            for i in range(len(pts) - 1):
                cv2.line(img_display, pts[i], pts[i + 1], (0, 255, 0), 2)
            cv2.circle(img_display, pts[-1], 3, (0, 0, 255), -1)

        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF

        # 'n' = Extract color histogram from this specific region
        if key == ord("n") and len(pts) > 2:
            # Create a mask specifically for this polygon area
            roi_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, np.array([pts], dtype=np.int32), 255)

            # Calculate 2D Histogram for Hue and Saturation channels
            # Hue ranges 0-180, Saturation 0-255. We use 30 bins for H and 32 for S to avoid overfitting
            hist = cv2.calcHist(
                [hsv_first], [0, 1], roi_mask, [30, 32], [0, 180, 0, 255]
            )

            # Normalize the histogram so it acts as a probability map (values 0 to 255)
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

            trained_histograms.append(hist)
            current_polygons.append(pts)
            print(f"Feature profile {len(trained_histograms)} trained successfully!")
            pts = []

        # 'm' = Start the automated batch processing loop
        elif key == ord("m"):
            if len(trained_histograms) == 0:
                print("Please train at least one region using 'n' first!")
                continue
            break

    cv2.destroyAllWindows()

    # Histogram Backprojection
    print(
        f"\n--> Executing Backprojection on {len(img_paths)} frames using {len(trained_histograms)} models..."
    )

    for img_path in img_paths:
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Initialize an empty mask for the entire frame
        master_prob_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Run backprojection for every trained color feature and combine them
        for hist in trained_histograms:
            # Backprojection creates a grayscale image where pixel brightness = probability of matching the feature
            prob_map = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 255], 1)
            master_prob_mask = cv2.bitwise_or(master_prob_mask, prob_map)

        # Threshold the combined probability map to get a clean binary mask
        # A threshold value of 40-50 ensures low-probability background matches are dropped
        _, final_mask = cv2.threshold(master_prob_mask, 50, 255, cv2.THRESH_BINARY)

        # Traditional morphological cleanup to eliminate isolated noise speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(
            final_mask, cv2.MORPH_CLOSE, kernel
        )  # Fills tiny holes inside character
        final_mask = cv2.morphologyEx(
            final_mask, cv2.MORPH_OPEN, kernel
        )  # Removes small background specks

        # Apply the mask
        final_result = cv2.bitwise_and(img, img, mask=final_mask)

        # Save frame
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, final_result)

    print(f"Success! Character pipeline finished. Assets exported to: {output_dir}")
