"""
Prompts user to manually crop images in a given directory for template creation.
"""

import cv2
from pathlib import Path
from dataclasses import dataclass
import argparse

WINDOW_NAME = "image to crop"

image = None
display = None
is_drawing = False
start_point = (0, 0)
end_point = (0, 0)


@dataclass(slots=True)
class Point:
    """
    2D point
    """

    x: int = -1
    y: int = -1


@dataclass(slots=True)
class Rect:
    """
    2D rect defined by two adjacent corner points
    """

    p1: Point = (-1, -1)
    p2: Point = (-1, -1)


def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events to draw rectangle.

    Args:
        event (any): type of event
        x (x): x mouse coord
        y (int): y mouse coord
        flags (any):
        param (any):
    """

    global is_drawing, start_point, end_point, display

    # Mouse button pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        display = image.copy()
        start_point = (x, y)
        end_point = (x, y)

    # Mouse moving while pressed
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:

        end_point = (x, y)

        # redraw image each frame
        display = image.copy()
        cv2.rectangle(display, start_point, end_point, (0, 255, 0), 1)

    # Mouse button released
    elif event == cv2.EVENT_LBUTTONUP:

        is_drawing = False
        end_point = (x, y)

        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])

        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        rect = Rect(Point(x1, y1), Point(x2, y2))

        cv2.rectangle(display, start_point, end_point, (0, 255, 0), 1)
    

        # display = image.copy()


def get_annotated_file_path(original_path: str) -> str:
    """
    Returns image path with _annotated suffix added to name.

    Args:
        original_path (str): Path of original image file

    Returns:
        str: File path for annotated image
    """
    annotated_path = Path(original_path)
    name_parts = annotated_path.name.split(".")
    annotated_file_type = name_parts[-1]
    name_parts = name_parts[:-1]
    annotated_file_name = "".join(name_parts) + "_annotated" + "." + annotated_file_type
    return str(annotated_path.parent / annotated_file_name)


if __name__ == "__main__":

    # Parse cli arguments and handle bad inputs
    parser = argparse.ArgumentParser(
        description="Label pixel regions in an image and save to csv.",
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
        raise FileNotFoundError(f"Input directory is not a valid directory")

    output_path = Path(args["output"]).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = [f for f in image_path.rglob("*") if f.is_file()]
    for f in input_files:
            
        image = cv2.imread(f)
        if image is None:
            raise RuntimeError("Input file is not a valid image.")

        # Create image window and display image
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME, image)

        # Register mouse event callback to handle drawing selection rect
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        # Main loop
        display = image.copy()
        while True:
            cv2.imshow(WINDOW_NAME, display)

            k = cv2.waitKey(1)
            if k == ord('x'):
                break
            if k == ord(" "):
                x = start_point[0] + 1
                y = start_point[1] + 1
                w = end_point[0] - x
                h = end_point[1] - y
                
                display = display[y: y + h, x: x + w]
                # Save cropped image to new image file
                cv2.imwrite(str(output_path / f.stem ) + ".jpg", display)
                break
        
    cv2.destroyAllWindows()

