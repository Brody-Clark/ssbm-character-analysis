import argparse
import json
import statistics
from typing import List, Tuple, Dict, Optional


def iou(rect_a: List[int], rect_b: List[int]) -> float:
    """Compute IoU for rects in [x,y,w,h] format."""
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_results(results_path: str) -> Dict[int, List[Dict]]:
    """Load pipeline results (jsonl) and return map frame_index -> record dict.

    Each value is a dict with keys: `actors` (list), optional `timestamp_s` (float),
    and optional `elapsed_frame_time_s` (float).
    """
    frames: Dict[int, Dict] = {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            fi = int(data.get("frame_index", 0))
            actors = data.get("actors", []) or []
            record: Dict = {"actors": actors}
            if "timestamp_s" in data:
                record["timestamp_s"] = data.get("timestamp_s")
            if "elapsed_frame_time_s" in data:
                record["elapsed_frame_time_s"] = data.get(
                    "elapsed_frame_time_s"
                )
            frames[fi] = record
    return frames


def load_ground_truth(gt_path: str) -> Dict[int, List[Dict]]:
    """Load ground truth JSON file (expected dict with frame_XXXX keys)."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames: Dict[int, List[Dict]] = {}
    for k, v in data.items():
        # Support either numeric keys or 'frame_0001' style
        frame_num = v.get("frame_number") if isinstance(v, dict) else None
        if frame_num is None:
            try:
                frame_num = int(k.replace("frame_", ""))
            except Exception:
                continue
        actors = v.get("actors", []) if isinstance(v, dict) else []
        frames[int(frame_num)] = actors
    return frames


def match_frame(
    gt_actors: List[Dict], pred_actors: List[Dict], iou_thresh: float = 0.5
) -> Tuple[int, int, int, List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """Match predictions to ground truth for a single frame.

    Returns: TP, FP, FN, list of matched pairs (gt,pred), unmatched_preds, unmatched_gts
    """
    if not gt_actors and not pred_actors:
        return 0, 0, 0, [], [], []

    gt_boxes = [g.get("bounding_rect") for g in gt_actors if g.get("bounding_rect") is not None]
    pred_boxes = [p.get("rect") for p in pred_actors]

    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    matches: List[Tuple[Dict, Dict]] = []

    # Build IoU matrix
    iou_mat = []
    for gi, gb in enumerate(gt_boxes):
        row = []
        for pj, pb in enumerate(pred_boxes):
            row.append(iou(gb, pb))
        iou_mat.append(row)

    # Greedy matching by highest IoU
    while True:
        best_val = 0.0
        best_g = -1
        best_p = -1
        for gi in range(len(gt_boxes)):
            if gt_matched[gi]:
                continue
            for pj in range(len(pred_boxes)):
                if pred_matched[pj]:
                    continue
                val = iou_mat[gi][pj]
                if val > best_val:
                    best_val = val
                    best_g = gi
                    best_p = pj
        if best_val >= iou_thresh and best_g >= 0 and best_p >= 0:
            gt_matched[best_g] = True
            pred_matched[best_p] = True
            matches.append((gt_actors[best_g], pred_actors[best_p]))
        else:
            break

    tp = len(matches)
    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)
    unmatched_preds = [pred_actors[i] for i, m in enumerate(pred_matched) if not m]
    unmatched_gts = [gt_actors[i] for i, m in enumerate(gt_matched) if not m]
    return tp, fp, fn, matches, unmatched_preds, unmatched_gts


def analyze(results_path: str, gt_path: str, iou_thresh: float = 0.5):
    results = load_results(results_path)
    gt = load_ground_truth(gt_path)

    all_frames = sorted(set(results.keys()) | set(gt.keys()))

    total_tp = total_fp = total_fn = 0
    total_gt = 0
    # classification broken into two metrics: character vs animation
    char_correct = 0
    anim_correct = 0
    class_total = 0
    idsw = 0

    # FPS tracking
    elapsed_times: List[float] = []
    timestamps: List[float] = []

    # Track assignment of GT actor_id -> predicted track_id
    prev_assignment: Dict[int, Optional[object]] = {}

    def split_pred_label(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not label:
            return None, None
        label = label.strip()
        if label.lower() == "unknown":
            return None, None
        if "_" in label:
            parts = label.split("_", 1)
            return parts[0], parts[1]
        return label, None


    for fi in all_frames:
        gt_actors = gt.get(fi, [])
        pred_record = results.get(fi, {}) or {}
        pred_actors = pred_record.get("actors", [])
        # collect timing if available
        if "elapsed_frame_time_s" in pred_record and isinstance(
            pred_record.get("elapsed_frame_time_s"), (int, float)
        ):
            elapsed_times.append(pred_record.get("elapsed_frame_time_s"))
        if "timestamp_s" in pred_record and isinstance(
            pred_record.get("timestamp_s"), (int, float)
        ):
            timestamps.append(pred_record.get("timestamp_s"))

        total_gt += len(gt_actors)

        tp, fp, fn, matches, unmatched_preds, unmatched_gts = match_frame(
            gt_actors, pred_actors, iou_thresh=iou_thresh
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Classification accuracy: split predicted label into character and animation
        for g, p in matches:
            pred_label = p.get("character_id")
            gt_labels = g.get("labels", [])
            pred_char, pred_anim = split_pred_label(pred_label)
            if pred_char and pred_char in gt_labels:
                char_correct += 1
            if pred_anim and pred_anim in gt_labels:
                anim_correct += 1
            class_total += 1

        # ID switch: use stable predicted track_id from the results
        for g, p in matches:
            gt_id = g.get("actor_id")
            if gt_id is None:
                continue
            pred_track_id = p.get("track_id")
            prev = prev_assignment.get(gt_id)
            if prev is None:
                # first observation
                prev_assignment[gt_id] = pred_track_id
            else:
                if pred_track_id is not None and pred_track_id != prev:
                    # Count switch only when the predicted track identity changes
                    idsw += 1
                    prev_assignment[gt_id] = pred_track_id

        # For GT actors not matched this frame, set their prev_assignment to None (missed)
        for g in unmatched_gts:
            gid = g.get("actor_id")
            if gid is not None:
                prev_assignment[gid] = None

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    char_accuracy = char_correct / class_total if class_total > 0 else 0.0
    anim_accuracy = anim_correct / class_total if class_total > 0 else 0.0

    # FPS metrics: processing FPS (based on elapsed_frame_time_s) and observed FPS (timestamps)
    proc_fps = obs_fps = 0.0
    mean_elapsed = median_elapsed = 0.0
    if elapsed_times:
        total_elapsed = sum(elapsed_times)
        proc_fps = len(elapsed_times) / total_elapsed if total_elapsed > 0 else 0.0
        mean_elapsed = statistics.mean(elapsed_times)
        try:
            median_elapsed = statistics.median(elapsed_times)
        except Exception:
            median_elapsed = mean_elapsed

    if len(timestamps) >= 2:
        timestamps_sorted = sorted(timestamps)
        span = timestamps_sorted[-1] - timestamps_sorted[0]
        if span > 0:
            obs_fps = (len(timestamps_sorted) - 1) / span

    mota = 1.0 - (total_fn + total_fp + idsw) / total_gt if total_gt > 0 else 0.0

    print("Evaluation Summary:")
    print(f"Frames evaluated        : {len(all_frames)}")
    print(f"Ground-truth objects    : {total_gt}")
    print(f"TP                      : {total_tp}")
    print(f"FP                      : {total_fp}")
    print(f"FN                      : {total_fn}")
    print(f"Precision               : {precision:.4f}")
    print(f"Recall                  : {recall:.4f}")
    print(f"F1                      : {f1:.4f}")
    print(f"Character accuracy      : {char_accuracy:.4f} ({char_correct}/{class_total})")
    print(f"Animation accuracy      : {anim_accuracy:.4f} ({anim_correct}/{class_total})")
    if elapsed_times:
        print(f"Avg elapsed/frame (s)   : {mean_elapsed:.4f}")
        print(f"Median elapsed/frame (s): {median_elapsed:.4f}")
        print(f"Processing FPS (frames/total_time): {proc_fps:.2f}")
    if obs_fps:
        print(f"Observed FPS (timestamps): {obs_fps:.2f}")
    print(f"ID switches (approx)    : {idsw}")
    print(f"MOTA (approx)           : {mota:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze pipeline results against ground truth"
    )
    parser.add_argument(
        "--results", required=True, help="Path to results.jsonl produced by pipeline"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth JSON file for the recording",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for detection matching (default 0.5)",
    )
    args = parser.parse_args()

    res_path = args.results
    gt_path = args.ground_truth
    analyze(res_path, gt_path, iou_thresh=args.iou)
