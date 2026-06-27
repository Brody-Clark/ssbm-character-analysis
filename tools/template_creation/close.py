"""
Closes images in a given directory for template creation.
"""

import cv2
from pathlib import Path
import argparse

if __name__ == "__main__":

    # Parse cli arguments and handle bad inputs
    parser = argparse.ArgumentParser(
        description="Close images for template creation.",
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

    image_path = Path(args["input"]).resolve()

    if not image_path.is_dir():
        raise FileNotFoundError("Input directory is not a valid directory")

    output_path = Path(args["output"]).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    input_files = [f for f in image_path.rglob("*") if f.is_file()]
    count = 0
    for f in input_files:

        image = cv2.imread(f)
        if image is None:
            print(f"Input file {f} is not a valid image. Skipping.")
            continue

        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)

        if not cv2.imwrite(str(output_path / f.stem) + ".jpg", image):
            print(f"Failed to write {f} to file. Skipping.")
            continue

        count = count + 1

    print(f"Close operations finished. Wrote {count} closed images to {str(output_path)}.")
