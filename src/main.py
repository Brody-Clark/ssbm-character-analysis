import argparse
from pathlib import Path
import logging
from contextlib import ExitStack
import sys
import json
from dataclasses import asdict
from cv2 import imread
import numpy as np
from ssbmca.core.pipeline import VisionPipeline
from ssbmca.core.feature_extractor import FeatureExtractor
from ssbmca.core.detector import Detector
from ssbmca.core.tracker import Tracker
from ssbmca.core.matcher import Matcher
from ssbmca.domain.models import FeatureDatabase, ActorFeatures
from ssbmca.source.frame_source import VideoSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_character_spritesheets(
    root_path: str | Path, extractor: FeatureExtractor
) -> ActorFeatures:
    """Load character sprite sheets from a directory of per-character animations."""
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid root path: {root_path}")

    actor_sprites = ActorFeatures()
    image_files: list[Path] = []
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir():
            continue
        for anim_dir in sorted(char_dir.iterdir()):
            if not anim_dir.is_dir():
                continue

            image_files.extend(
                [
                    path
                    for path in anim_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
                ]
            )

    features_list = []
    try:
        for img_path in image_files:
            img = imread(str(img_path))
            if img is None:
                continue
            actor_sprites.character_names.append(img_path.parent.parent.name)
            actor_sprites.animation_names.append(img_path.parent.name)
            success, features = extractor.get_character_features(img)
            if not success:
                _logger.warning(
                    "Failed to extract features from %s. Skipping.", img_path
                )
                continue
            features_list.append(features)
        actor_sprites.features = np.array(features_list, dtype=np.float32)
    except Exception:
        _logger.error("Failed to load extract features from templates", exc_info=True)
        sys.exit(1)

    return actor_sprites


def _numpy_json_encoder(obj):
    """Convert NumPy arrays and scalars to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def handle_init(args, parser):
    """Generates feature index file from provided templates"""
    output_file = Path(args.output)
    if output_file.suffix.lower() != ".json":
        parser.error("The output file must have a .json extension.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sprite_path = Path(args.sprites_dir)
    if not sprite_path.is_absolute():
        sprite_path = (PROJECT_ROOT / sprite_path).resolve()
    if not sprite_path.exists():
        raise RuntimeError(f"Invalid sprite data root path {sprite_path}")

    _logger.info("Generating feature index.")
    extractor = FeatureExtractor(args.features)
    actor_sprites = _load_character_spritesheets(
        sprite_path / "sprites", extractor=extractor
    )

    db = FeatureDatabase(actor_features=actor_sprites)

    with open(str(output_file), "w", encoding="utf8") as file:
        json.dump(asdict(db), file, default=_numpy_json_encoder, indent=2)

    _logger.info(
        "Initialization complete. Index file created at %s", str(output_file.absolute())
    )


def _load_feature_database(json_path: str | Path) -> FeatureDatabase:
    """Load and parse a SpriteDatabase from a JSON index file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actor_data = data.get("actor_features", {})

    actor_features = ActorFeatures(
        character_names=actor_data.get("character_names", []),
        animation_names=actor_data.get("animation_names", []),
        features=np.array(actor_data.get("features", []), dtype=np.float32),
    )

    return FeatureDatabase(actor_features=actor_features)


def handle_run(args, parser):
    """Loads index and runs pipeline"""
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    if not input_path.exists():
        parser.error(f"Input video not found: {input_path}")

    index = Path(args.index)
    if not input_path.exists():
        parser.error(f"Index file not found: {index}")

    if args.output == "file" and not args.file:
        parser.error("--file is required when --output is set to 'file'.")

    _logger.info("Loading index file.")
    try:
        sprite_db = _load_feature_database(index)
    except FileNotFoundError:
        parser.error("The specified index file does not exist.")
    except json.JSONDecodeError:
        parser.error("The index file contains invalid JSON syntax.")
    except KeyError:
        parser.error("The index JSON file is invalid or corrupt.")

    with ExitStack() as stack:
        # Determine the output stream
        if args.output == "file":
            output_file = Path(args.file)
            if output_file.suffix.lower() != ".jsonl":
                parser.error("The output file must have a .jsonl extension.")
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                out_stream = stack.enter_context(open(args.file, "w", encoding="utf-8"))
            except OSError as e:
                print(f"Error opening file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            out_stream = sys.stdout

        _logger.info("Initializing pipeline.")

        frame_source = VideoSource(video_frame_source=str(input_path))
        detector = Detector(stage_name=args.stage)
        tracker = Tracker()
        matcher = Matcher(
            feature_extractor=FeatureExtractor(features=args.features),
            feature_database=sprite_db,
        )
        pipeline = VisionPipeline(detector=detector, tracker=tracker, matcher=matcher)

        _logger.info("Initialization complete, beginning processing.")

        # Process video
        pipeline.process(
            video_source=frame_source, output_stream=out_stream, debug=args.debug
        )
        _logger.info("Processing complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available subcommands"
    )
    init_parser = subparsers.add_parser("init", help="Initialize template database.")
    init_parser.add_argument(
        "--sprites-dir",
        "-s",
        type=str,
        default="./data/templates",
        help="Path to sprite template directory.",
    )
    init_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to store generate sprite template database .json.",
    )
    init_parser.add_argument(
        "--features",
        "-f",
        nargs="+",
        choices=["fd", "hu", "sat", "hsv", "lbp"],
        help="Choose one or more features for classification.",
    )
    init_parser.set_defaults(func=handle_init)

    run_parser = subparsers.add_parser(
        "run", help="Run SSBM Character Analysis pipeline."
    )

    run_parser.add_argument(
        "--input", "-i", required=True, type=str, help="Path to video file to analyze."
    )
    run_parser.add_argument(
        "--output",
        choices=["stdout", "file"],
        default="stdout",
        help="Choose where to send the output (default: stdout)",
    )
    run_parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to the template index .json file created by running the init command.",
    )
    run_parser.add_argument(
        "--file",
        type=str,
        help="Path to the output file (required if --output is set to 'file')",
    )
    run_parser.add_argument(
        "--stage",
        "-s",
        required=True,
        help="name of stage.",
        choices=["temple", "corneria", "venom"],
    )
    run_parser.add_argument(
        "--features",
        "-f",
        nargs="+",
        choices=["fd", "hu", "sat", "hsv", "lbp"],
        help="Choose one or more features for classification. These must match what was used with the init command.",
    )
    run_parser.add_argument("--debug", action="store_true")
    run_parser.set_defaults(func=handle_run)

    args = parser.parse_args()
    args.func(args, parser)
