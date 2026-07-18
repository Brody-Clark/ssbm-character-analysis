import cv2
import os
import shutil
import numpy as np

INPUT_DIR = "D:\\Dev\\Super Smash Bros Vision\\data\\frames\\mario\\cropped"  
OUTPUT_DIR = "D:\\Dev\\Super Smash Bros Vision\\data\\frames\\mario\\cropped\\labeled"     

ANIMATION_MAP = {
    '0': 'idle',
    '1': 'walk',
    '2': 'run',
    '3': 'jump',
    '4': 'attack',
    '5': 'attack_side',
    '6': 'attack_up',
    '7': 'special',
    '8': 'special_side',
    '9': 'special_up',
    'a': 'special_down',
    'b': 'damaged',
    'c': 'guard',
    'd': 'crouch',
    'e': 'attack_down',
    'f': 'fall',
    'g': 'recover',
    'h': 'dodge'
}

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for folder_name in ANIMATION_MAP.values():
        path = os.path.join(OUTPUT_DIR, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)

def create_dashboard(image_path, current_input, current_idx, total_images):
    # Load raw image asset
    img = cv2.imread(image_path)
    if img is None:
        # Fallback dummy canvas if image read error occurs
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
    # Standardize image display canvas dimension height
    canvas_h = 600
    aspect_ratio = img.shape[1] / img.shape[0]
    new_w = int(canvas_h * aspect_ratio)
    img_resized = cv2.resize(img, (new_w, canvas_h))
    
    # Generate structural text-info layout board sidebar (420px width)
    sidebar_w = 420
    sidebar = np.zeros((canvas_h, sidebar_w, 3), dtype=np.uint8)
    sidebar[:] = (30, 30, 30) # Soft dark background gray
    
    # Text placement variables
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_spacing = 24
    
    # Draw Progress Header
    cv2.putText(sidebar, f"Image: {current_idx + 1}/{total_images}", (15, 30), font, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    # cv2.putText(sidebar, f"File: {os.path.basename(image_path)}", (15, 55), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Draw Divider Line
    cv2.line(sidebar, (15, 65), (sidebar_w - 15, 65), (70, 70, 70), 1)
    
    # Render Animation Map Menu Options
    y_offset = 90
    for key, label in ANIMATION_MAP.items():
        # Highlight logic: turn cyan if selected in active buffer
        is_selected = (current_input == key)
        color = (235, 206, 135) if is_selected else (170, 170, 170)
        thick = 2 if is_selected else 1
        prefix = "-> " if is_selected else "   "
        
        menu_text = f"{prefix}[{key}] : {label}"
        cv2.putText(sidebar, menu_text, (15, y_offset), font, font_scale, color, thick, cv2.LINE_AA)
        y_offset += line_spacing
        
    # Draw Live Selection Status Bar at bottom
    cv2.line(sidebar, (15, canvas_h - 80), (sidebar_w - 15, canvas_h - 80), (70, 70, 70), 1)
    
    status_label = ANIMATION_MAP.get(current_input, "NONE")
    cv2.putText(sidebar, f"Buffer: [{current_input}] -> {status_label}", (15, canvas_h - 50), font, 0.55, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(sidebar, "[Space]: Confirm & Save  |  [Esc]: Exit", (15, canvas_h - 20), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    
    # Combine resized sprite crop window beside text layout
    combined_window = np.hstack((img_resized, sidebar))
    return combined_window

def main():
    setup_directories()
    
    # Retrieve valid image targets
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    images.sort()
    
    if not images:
        print(f"Error: No image assets detected inside directory '{INPUT_DIR}'")
        return
        
    cv2.namedWindow("Melee CV Annotation Tool", cv2.WINDOW_AUTOSIZE)
    
    idx = 0
    current_buffer = ""
    total_imgs = len(images)
    
    while idx < total_imgs:
        img_path = images[idx]
        
        # Display render frame loop
        display_frame = create_dashboard(img_path, current_buffer, idx, total_imgs)
        cv2.imshow("Melee CV Annotation Tool", display_frame)
        
        # Pull keyboard interaction codes
        key = cv2.waitKey(0) & 0xFF
        char_key = chr(key)
        
        # Escape sequence breaks app
        if key == 27: 
            print("Session terminated manually by user.")
            break
            
        # If character pressed is in our key dictionary, store it in buffer
        elif char_key in ANIMATION_MAP:
            current_buffer = char_key
            
        # Spacebar confirms execution and moves item to target class folder
        elif key == 32:
            if current_buffer in ANIMATION_MAP:
                assigned_class = ANIMATION_MAP[current_buffer]
                target_folder = os.path.join(OUTPUT_DIR, assigned_class)
                
                # Create a unique incremental naming format for file tracking stability
                existing_count = len(os.listdir(target_folder))
                file_extension = os.path.splitext(img_path)[1]
                new_filename = f"{assigned_class}_{existing_count:04d}{file_extension}"
                destination_path = os.path.join(target_folder, new_filename)
                
                # Copy instead of move to preserve your source recording folder integrity
                shutil.copy(img_path, destination_path)
                
                # Advance iteration tracking variables
                idx += 1
                current_buffer = "" # Flush active key buffer string
            else:
                # Trigger a system window flash warn sound if pressing space with no assignments loaded
                print("Warning: Load a valid numeric option before pressing Spacebar.")

    cv2.destroyAllWindows()
    print("Labeling complete!")

if __name__ == '__main__':
    # Ensure source path folder matches target layout setups before executing
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Created initial input directory '{INPUT_DIR}'. Drop your 450 frames inside here and restart.")
    else:
        main()