import argparse
import os
from pathlib import Path
import cv2


def expand_bbox(
    x: int, y: int, w: int, h: int, scale: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Expands bounding box by a percentage scale factor while clamping to image borders."""
    dw = int(w * (scale - 1.0) / 2)
    dh = int(h * (scale - 1.0) / 2)

    x1 = max(0, x - dw)
    y1 = max(0, y - dh)
    x2 = min(img_w, x + w + dw)
    y2 = min(img_h, y + h + dh)

    return x1, y1, x2, y2


def process_images(input_dir: str, output_dir: str, scale: float = 1.20):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]

    if not image_files:
        print(f"No valid images found in: {input_path}")
        return

    print(f"Processing {len(image_files)} image(s)...")

    for file_path in image_files:
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Skipping unreadable file: {file_path.name}")
            continue

        img_h, img_w = img.shape[:2]

        # Convert to grayscale & binary threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Using OTSU thresholding handles varying brightness across images automatically
        _, binary = cv2.threshold(
            gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find external contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print(f"No foreground detected in: {file_path.name}")
            continue

        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get base bounding box and apply small padding expansion
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1, y1, x2, y2 = expand_bbox(x, y, w, h, scale, img_w, img_h)

        # Crop image to padded box and save
        cropped_img = img[y1:y2, x1:x2]
        out_file = output_path / file_path.name
        cv2.imwrite(str(out_file), cropped_img)

    print("Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Crop images around the largest contour with a small percent bounding box margin."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the output directory where cropped images will be saved.",
    )

    args = parser.parse_args()
    process_images(args.input, args.output)


if __name__ == "__main__":
    main()