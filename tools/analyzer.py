import json
import statistics
from pathlib import Path


def parse_json_lines(file_path):
    frame_times = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if "elapsed_frame_time_s" in data:
                        frame_times.append(data["elapsed_frame_time_s"])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    if not frame_times:
        print("No 'elapsed_frame_time_s' data found.")
        return

    total_frames = len(frame_times)
    avg_frame_time = statistics.mean(frame_times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    print("--- Pipeline Performance Summary ---")
    print(f"Total Frames Processed : {total_frames}")
    print(f"Average Frame Time     : {avg_frame_time * 1000:.2f} ms")
    print(f"Average FPS            : {avg_fps:.2f}")


if __name__ == "__main__":
    # Change to your actual file path
    # Set is_json_lines=False if your file wraps everything in brackets [...]
    file = Path.cwd() / "results.jsonl"
    parse_json_lines(str(file))
