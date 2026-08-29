import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

# Map feature combo names to CLI argument choices
FEATURE_CONFIGS: Dict[str, List[str]] = {
    "Color-Only": ["hsv"],
    "Shape-Only": ["fd", "hu"],
    "Texture+Shape": ["fd", "hu", "lbp"],
    "Color-Invariant": ["fd", "lbp", "sat"],
    "Texture+Shape+Color": ["fd", "lbp", "hsv"],
    "Shape+Color-1": ["fd", "hsv"],
    "Shape+Color-2": ["hu", "hsv"],
    "Full-Ensemble": ["fd", "hu", "lbp", "hsv"],
}

STAGES: List[str] = ["corneria", "venom", "temple"]
NUM_REPEATS: int = 3 # Runs for avg FPS


def run_command(cmd: List[str]) -> None:
    """Executes a subprocess command and raises an error if it fails."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nError: {result.stderr}")


def run_experiment_suite(
    main_script: str = "./src/main.py",
    analyzer_script: str = "./tools/analyze_results.py",
    video_dir: str = "./tests/recordings",
    ground_truth_dir: str = "./tests/labels",
    output_dir: str = "./test_results",
    sprites_dir: str = "./data/templates",
) -> List[Dict[str, Any]]:
    """Runs the init/run/analyze benchmark pipeline across all feature sets and stages."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_test_metrics: List[Dict[str, Any]] = []

    for config_name, features in FEATURE_CONFIGS.items():
        print(f"-- Running Configuration: {config_name} ({features}) --")

        # Init Step: Produce unique index for feature combo
        index_file = out_path / f"index_{config_name}.json"
        init_cmd = [
            sys.executable,
            main_script,
            "init",
            "--sprites-dir", sprites_dir,
            "--output", str(index_file),
            "--features", *features,
        ]
        print(f"Initializing templates database: {index_file.name}")
        run_command(init_cmd)

        # Run & Analyze per Stage
        for stage in STAGES:
            video_file = f"{video_dir}/{stage}.mp4"
            gt_file = f"{ground_truth_dir}/{stage}.json"

            stage_run_metrics: List[Dict[str, Any]] = []

            for run_idx in range(1, NUM_REPEATS + 1):
                results_jsonl = out_path / f"results_{config_name}_{stage}_run{run_idx}.jsonl"
                metrics_json = out_path / f"metrics_{config_name}_{stage}_run{run_idx}.json"

                # Run main.py pipeline
                run_cmd = [
                    sys.executable,
                    main_script,
                    "run",
                    "--input", video_file,
                    "--stage", stage,
                    "--index", str(index_file),
                    "--output", "file",
                    "--file", str(results_jsonl),
                    "--features", *features,
                ]
                print(f"    Executing [{stage}] Run {run_idx}/{NUM_REPEATS}...")
                run_command(run_cmd)

                # Call analyzer script to output metrics JSON
                analyze_cmd = [
                    sys.executable,
                    analyzer_script,
                    "--results", str(results_jsonl),
                    "--ground-truth", gt_file,
                    "--output", str(metrics_json),
                ]
                run_command(analyze_cmd)

                # Read run result JSON for aggregation
                with open(metrics_json, "r", encoding="utf-8") as f:
                    stage_run_metrics.append(json.load(f))

            # Aggregate 3x runs per Stage for avg FPS measure
            avg_fps = sum(m["performance"]["avg_fps"] for m in stage_run_metrics) / NUM_REPEATS
            mean_frame_time = sum(m["performance"]["avg_time_per_frame_s"] for m in stage_run_metrics) / NUM_REPEATS

            # Pick stable metrics from run 1 since metrics don't change
            base_metrics = stage_run_metrics[0]

            aggregated_record = {
                "config_name": config_name,
                "features": features,
                "stage": stage,
                "detection_f1": base_metrics["detection"]["f1"],
                "mota": base_metrics["tracking"]["mota"],
                "character_accuracy": base_metrics["classification"]["character_accuracy"],
                "animation_accuracy": base_metrics["classification"]["animation_accuracy"],
                "avg_fps": round(avg_fps, 2),
                "avg_time_per_frame_s": round(mean_frame_time, 4),
            }

            all_test_metrics.append(aggregated_record)
            print(f"    Stage [{stage}] Complete | Avg FPS: {avg_fps:.2f} | F1: {base_metrics['detection']['f1']}")

    # Save aggregated summary file for all runs
    summary_file = out_path / "aggregated_test_suite_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_test_metrics, f, indent=2)

    print(f"\nSuite complete. Summary written to {summary_file}")
    return all_test_metrics


if __name__ == "__main__":
    run_experiment_suite()