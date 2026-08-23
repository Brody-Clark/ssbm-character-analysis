import argparse
import json
import statistics
from typing import List, Tuple, Dict, Optional, Any


def iou(rect_a: List[int], rect_b: List[int]) -> float:
    """Compute IoU for openCV rects."""
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


def load_results(results_path: str) -> Dict[int, Dict]:
    """Load pipeline results from jsonl and return map frame_index -> record dict."""
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
                record["elapsed_frame_time_s"] = data.get("elapsed_frame_time_s")
            frames[fi] = record
    return frames


def load_ground_truth(gt_path: str) -> Dict[int, List[Dict]]:
    """Load ground truth JSON file (expected dict with frame_XXXX keys)."""
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames: Dict[int, List[Dict]] = {}
    for k, v in data.items():
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
    """Match predictions to ground truth for a single frame."""
    if not gt_actors and not pred_actors:
        return 0, 0, 0, [], [], []

    visible_gt_actors = [g for g in gt_actors if g.get("bounding_rect") is not None]
    gt_boxes = [g.get("bounding_rect") for g in visible_gt_actors]
    pred_boxes = [p.get("rect") for p in pred_actors]

    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    matches: List[Tuple[Dict, Dict]] = []

    iou_mat = []
    for gb in gt_boxes:
        row = []
        for pb in pred_boxes:
            row.append(iou(gb, pb))
        iou_mat.append(row)

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
            matches.append((visible_gt_actors[best_g], pred_actors[best_p]))
        else:
            break

    tp = len(matches)
    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)
    unmatched_preds = [pred_actors[i] for i, m in enumerate(pred_matched) if not m]
    unmatched_gts = [visible_gt_actors[i] for i, m in enumerate(gt_matched) if not m]
    return tp, fp, fn, matches, unmatched_preds, unmatched_gts


def analyze(results_path: str, gt_path: str, iou_thresh: float = 0.5) -> Dict[str, Any]:
    """Compares ground truth to predicted results and returns structured evaluation metrics."""
    results = load_results(results_path)
    gt = load_ground_truth(gt_path)

    all_frames = sorted(set(results.keys()) | set(gt.keys()))

    total_tp = total_fp = total_fn = 0
    total_gt = 0

    char_correct = 0
    anim_correct = 0
    class_total = 0
    idsw = 0

    elapsed_times: List[float] = []
    prev_assignment: Dict[int, Optional[object]] = {}

    for fi in all_frames:
        gt_actors = gt.get(fi, [])
        visible_gt_actors = [g for g in gt_actors if g.get("bounding_rect") is not None]
        pred_record = results.get(fi, {}) or {}
        pred_actors = pred_record.get("actors", [])

        if "elapsed_frame_time_s" in pred_record and isinstance(
            pred_record.get("elapsed_frame_time_s"), (int, float)
        ):
            elapsed_times.append(pred_record.get("elapsed_frame_time_s"))

        total_gt += len(visible_gt_actors)

        tp, fp, fn, matches, _, _ = match_frame(
            visible_gt_actors, pred_actors, iou_thresh=iou_thresh
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for g, p in matches:
            pred_char = p.get("character_id")
            pred_anim = p.get("animation_id")
            gt_labels = g.get("labels", [])
            if pred_char and pred_char in gt_labels:
                char_correct += 1
            if pred_anim and pred_anim in gt_labels:
                anim_correct += 1
            class_total += 1

        for g, p in matches:
            gt_id = g.get("actor_id")
            if gt_id is None:
                continue
            pred_track_id = p.get("track_id")
            prev = prev_assignment.get(gt_id)
            if prev is not None and pred_track_id is not None and pred_track_id != prev:
                idsw += 1

            if pred_track_id is not None:
                prev_assignment[gt_id] = pred_track_id

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    matching_accuracy = (
        total_tp / (total_tp + total_fp + total_fn)
        if (total_tp + total_fp + total_fn) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    char_accuracy = char_correct / class_total if class_total > 0 else 0.0
    anim_accuracy = anim_correct / class_total if class_total > 0 else 0.0
    mota = 1.0 - (total_fn + total_fp + idsw) / total_gt if total_gt > 0 else 0.0

    proc_fps = mean_elapsed = median_elapsed = 0.0
    if elapsed_times:
        total_elapsed = sum(elapsed_times)
        proc_fps = len(elapsed_times) / total_elapsed if total_elapsed > 0 else 0.0
        mean_elapsed = statistics.mean(elapsed_times)
        try:
            median_elapsed = statistics.median(elapsed_times)
        except Exception:
            median_elapsed = mean_elapsed

    return {
        "frames_evaluated": len(all_frames),
        "detection": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "segmentation_accuracy": round(matching_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "tracking": {
            "visible_gt_objects": total_gt,
            "id_switches": idsw,
            "mota": round(mota, 4),
        },
        "classification": {
            "character_accuracy": round(char_accuracy, 4),
            "character_correct": char_correct,
            "animation_accuracy": round(anim_accuracy, 4),
            "animation_correct": anim_correct,
            "total_matches": class_total,
        },
        "performance": {
            "avg_time_per_frame_s": round(mean_elapsed, 4),
            "median_time_per_frame_s": round(median_elapsed, 4),
            "avg_fps": round(proc_fps, 2),
        },
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    """Helper to format and print human-readable summary to console."""
    det = metrics["detection"]
    trk = metrics["tracking"]
    cls = metrics["classification"]
    prf = metrics["performance"]

    print("Evaluation Summary:")
    print(f"Frames evaluated        : {metrics['frames_evaluated']}")
    print("Detection:")
    print(f"TP                      : {det['tp']}")
    print(f"FP                      : {det['fp']}")
    print(f"FN                      : {det['fn']}")
    print(f"Segmentation accuracy   : {det['segmentation_accuracy']:.4f}")
    print(f"Precision               : {det['precision']:.4f}")
    print(f"Recall                  : {det['recall']:.4f}")
    print(f"F1                      : {det['f1']:.4f}")
    print("\nTracking:")
    print(f"Visible GT objects      : {trk['visible_gt_objects']}")
    print(f"ID switches (approx)    : {trk['id_switches']}")
    print(f"MOTA (approx)           : {trk['mota']:.4f}")
    print("\nClassification:")
    print(
        f"Character accuracy      : {cls['character_accuracy']:.4f} ({cls['character_correct']}/{cls['total_matches']})"
    )
    print(
        f"Animation accuracy      : {cls['animation_accuracy']:.4f} ({cls['animation_correct']}/{cls['total_matches']})"
    )
    print("\nPerformance:")
    print(f"Avg time/frame (s)      : {prf['avg_time_per_frame_s']:.4f}")
    print(f"Median time/frame (s)   : {prf['median_time_per_frame_s']:.4f}")
    print(f"Average FPS             : {prf['avg_fps']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pipeline results against ground truth"
    )
    parser.add_argument(
        "--results", required=True, help="Path to results.jsonl produced by pipeline"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for detection matching (default 0.5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON to stdout instead of formatted text",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write JSON metrics file"
    )

    args = parser.parse_args()
    metrics = analyze(args.results, args.ground_truth, iou_thresh=args.iou)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_summary(metrics)


if __name__ == "__main__":
    main()
