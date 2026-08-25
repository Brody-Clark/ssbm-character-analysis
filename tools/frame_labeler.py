import cv2
import json
import os
import glob
from pathlib import Path

# --- CONFIGURATION ---
IMAGE_DIR = str(Path.cwd() / "data"  / "venom")    # Path to extracted FFMPEG frames
OUTPUT_JSON = str(Path.cwd() / "data" /  "recordings" /"venom_test.json")
NUM_ACTORS = 2              # 2 characters per frame

# --- HOTKEY MAPPING (Single Character -> Label) ---
LABEL_HOTKEYS = {
    '0': 'mario',
    '1': 'kirby',
    '2': 'idle',
    '3': 'walk',
    '4': 'run',
    '5': 'jump',
    '6': 'attack',
    '7': 'attack_side',
    '8': 'attack_up',
    '9': 'attack_down',
    'a': 'special',
    'b': 'special_side',
    'c': 'special_up',
    'd': 'special_down',
    'e': 'damaged',
    'f': 'guard',
    'g': 'crouch',
    'h': 'fall',
    'i': 'recover',
    'j': 'dodge',
}

# Mouse variables
drawing = False
ix, iy = -1, -1
current_rect = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect = (ix, iy, 0, 0)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_rect = (ix, iy, x - ix, y - iy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (ix, iy, x - ix, y - iy)

def normalize_rect(rect):
    """Normalize bounding boxes drawn in any direction (e.g. bottom-right to top-left)."""
    if rect is None:
        return None
    x, y, w, h = rect
    if w < 0:
        x += w
        w = abs(w)
    if h < 0:
        y += h
        h = abs(h)
    return [x, y, w, h]

def load_annotations():
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_annotations(data):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)

def run_pass_1_boxes(image_files, actor_idx):
    global current_rect
    cv2.namedWindow("Pass 1: Bounding Box")
    cv2.setMouseCallback("Pass 1: Bounding Box", draw_rectangle)
    
    boxes = {}
    print(f"\n--- PASS 1 (Actor {actor_idx + 1}): DRAW BOUNDING BOXES ---")
    print("Controls:\n - Mouse Drag: Draw Box\n - Spacebar: Save Box & Next Frame\n - R: Reset Box")

    for idx, filepath in enumerate(image_files):
        img = cv2.imread(filepath)
        if img is None:
            continue

        current_rect = None
        while True:
            display_img = img.copy()
            
            # Context overlays
            cv2.putText(display_img, f"Actor {actor_idx + 1} | Frame {idx+1}/{len(image_files)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_img, "[Drag]: Draw | [Space]: Save & Next | [R]: Reset Box", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if current_rect is not None:
                x, y, w, h = normalize_rect(current_rect)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Pass 1: Bounding Box", display_img)
            key = cv2.waitKey(20) & 0xFF

            if key == 32:  # Spacebar
                boxes[filepath] = normalize_rect(current_rect)
                break
            elif key in (ord('r'), ord('R')):
                current_rect = None

    cv2.destroyAllWindows()
    return boxes

def run_pass_2_labels(image_files, actor_idx, boxes):
    cv2.namedWindow("Pass 2: Labeling")
    annotations = {}

    print(f"\n--- PASS 2 (Actor {actor_idx + 1}): HOTKEY LABELING ---")
    print("Controls:\n - Press assigned Keys (0-9, a-j) to add labels\n - Backspace: Reset labels on current frame\n - Spacebar: Commit labels & Next Frame")

    for idx, filepath in enumerate(image_files):
        img = cv2.imread(filepath)
        if img is None:
            continue

        box = boxes.get(filepath)
        active_labels = []

        while True:
            display_img = img.copy()

            # Render Bounding Box
            if box and box[2] > 0 and box[3] > 0:
                x, y, w, h = box
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Render Header Info
            cv2.putText(display_img, f"Actor {actor_idx + 1} | Frame {idx+1}/{len(image_files)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Display Selected Labels
            selected_str = f"Selected: {', '.join(active_labels)}"
            cv2.putText(display_img, selected_str, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display Quick Reference HUD for Hotkeys
            y_offset = 100
            cv2.putText(display_img, "HOTKEYS: [Backspace]: Reset | [Space]: Save & Next", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            y_offset += 20
            # Render keybindings legend on screen
            for key_char, label in LABEL_HOTKEYS.items():
                legend_line = f"[{key_char}] -> {label}"
                cv2.putText(display_img, legend_line, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 16
                if y_offset > display_img.shape[0] - 20:  # Prevent overflowing bottom screen
                    break

            cv2.imshow("Pass 2: Labeling", display_img)
            key = cv2.waitKey(0) & 0xFF

            if key == 32:  # Spacebar -> Commit and move to next frame
                annotations[filepath] = {
                    "rect": box,
                    "labels": list(active_labels)
                }
                break

            elif key == 8:  # Backspace -> Reset active input on current frame
                active_labels.clear()

            else:
                # Convert pressed ASCII key to character
                char_pressed = chr(key).lower() if key < 128 else None
                
                # Check if char matches any registered hotkeys
                if char_pressed in LABEL_HOTKEYS:
                    label_to_add = LABEL_HOTKEYS[char_pressed]
                    if label_to_add not in active_labels:
                        active_labels.append(label_to_add)

    cv2.destroyAllWindows()
    return annotations

def main():
    valid_exts = ("*.png", "*.jpg", "*.jpeg")
    image_files = []
    for ext in valid_exts:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    image_files = sorted(image_files)

    if not image_files:
        print(f"Error: No images found in '{IMAGE_DIR}'. Please extract frames first.")
        return

    db = load_annotations()

    for actor_idx in range(NUM_ACTORS):
        print(f"\n==========================================")
        print(f"         PROCESSING ACTOR {actor_idx + 1}")
        print(f"==========================================")

        # Pass 1: Bounding boxes
        boxes = run_pass_1_boxes(image_files, actor_idx)

        # Pass 2: Quick Hotkey Labeling
        labels_data = run_pass_2_labels(image_files, actor_idx, boxes)

        # Merge into master JSON dataset
        for frame_num, filepath in enumerate(image_files):
            frame_key = f"frame_{frame_num + 1:04d}"
            
            if frame_key not in db:
                image_name = Path(filepath).name
                db[frame_key] = {
                    "frame_number": frame_num + 1,
                    "image_name": image_name,
                    "actors": []
                }

            actor_info = labels_data.get(filepath, {})
            actor_entry = {
                "actor_id": actor_idx + 1,
                "bounding_rect": actor_info.get("rect"),
                "labels": actor_info.get("labels", [])
            }

            if len(db[frame_key]["actors"]) > actor_idx:
                db[frame_key]["actors"][actor_idx] = actor_entry
            else:
                db[frame_key]["actors"].append(actor_entry)

        save_annotations(db)
        print(f"\nSuccessfully committed Actor {actor_idx + 1} data to '{OUTPUT_JSON}'.")

if __name__ == "__main__":
    main()