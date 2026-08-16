import argparse
from pathlib import Path
import logging
from contextlib import ExitStack
import sys
import json
from dataclasses import asdict
from cv2 import imread
import numpy as np
from ssbmv.core.pipeline import VisionPipeline
from ssbmv.core.feature_extractor import FeatureExtractor
from ssbmv.core.detector import Detector
from ssbmv.core.tracker import Tracker
from ssbmv.core.matcher import Matcher
from ssbmv.domain.models import SpriteDatabase, ActorSprites, HUDSprites
from ssbmv.source.frame_source import VideoSource
import cProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
_logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_character_huds(
    root_path: str | Path, extractor: FeatureExtractor
) -> HUDSprites:
    """Load HUD icon images from a flat directory of image files."""
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid root path: {root_path}")

    hud_sprites = HUDSprites()
    features_list = []
    for img_path in sorted(root.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
            _logger.debug("File %s is not valid. Skipping.", str(img_path))
            continue

        img = imread(str(img_path))
        if img is not None:
            hud_sprites.character_names.append(img_path.stem)
            success, features = extractor.get_character_features(img)
            if not success:
                _logger.warning(
                    "Failed to extract features from %s. Skipping.", img_path
                )
                continue
            features_list.append(features)
        hud_sprites.features = np.array(features_list, dtype=np.float32)

    return hud_sprites


def _load_character_spritesheets(
    root_path: str | Path, extractor: FeatureExtractor
) -> ActorSprites:
    """Load character sprite sheets from a directory of per-character animations."""
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid root path: {root_path}")

    actor_sprites = ActorSprites()
    image_files: list[Path] = []
    for char_dir in sorted(root.iterdir()):
        if not char_dir.is_dir():
            continue
        for anim_dir in sorted(char_dir.iterdir()):
            if not anim_dir.is_dir():
                continue

            image_files.extend([
                path
                for path in anim_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            ])

    features_list = (
        []
    )  # TODO: Figure out dimensions of features and use features = np.empty((num_images, feature_dim), dtype=np.float32)
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
    extractor = FeatureExtractor()
    actor_sprites = _load_character_spritesheets(
        sprite_path / "sprites", extractor=extractor
    )
    hud_sprites = _load_character_huds(sprite_path / "huds", extractor=extractor)

    db = SpriteDatabase(actor_sprites=actor_sprites, hud_sprites=hud_sprites)

    with open(str(output_file), "w", encoding="utf8") as file:
        json.dump(asdict(db), file, default=_numpy_json_encoder, indent=2)

    _logger.info(f"Initilization complete. Index file created at {str(output_file)}")


def _sprite_db_decoder(obj_dict):
    
    return SpriteDatabase(**obj_dict)

def _load_sprite_database(json_path: str | Path) -> SpriteDatabase:
    """Load and parse a SpriteDatabase from a JSON index file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actor_data = data.get("actor_sprites", {})
    hud_data = data.get("hud_sprites", {})

    actor_sprites = ActorSprites(
        character_names=actor_data.get("character_names", []),
        animation_names=actor_data.get("animation_names", []),
        features=np.array(actor_data.get("features", []), dtype=np.float32)
    )

    hud_sprites = HUDSprites(
        character_names=hud_data.get("character_names", []),
        features=np.array(hud_data.get("features", []), dtype=np.float32)
    )

    return SpriteDatabase(
        actor_sprites=actor_sprites,
        hud_sprites=hud_sprites
    )

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
        sprite_db = _load_sprite_database(index)
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
        matcher = Matcher(feature_extractor=FeatureExtractor(), sprite_database=sprite_db)
        pipeline = VisionPipeline(detector=detector, tracker=tracker, matcher=matcher)

        _logger.info("Initialization complete, beginning processing.")

        # Process video
        profiler = cProfile.Profile()
        profiler.enable()
        pipeline.process(
            video_source=frame_source, output_stream=out_stream, debug=args.debug
        )
        profiler.disable()
        # Save output to a .prof file in your workspace root
        profiler.dump_stats("pipeline.prof")
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
    init_parser.set_defaults(func=handle_init)

    run_parser = subparsers.add_parser("run", help="Run SSBMV pipeline.")

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
        help="Path to the template index .json file created by running the init command.",
        default="./index.json",
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
    run_parser.add_argument("--debug", action="store_true")
    run_parser.set_defaults(func=handle_run)

    args = parser.parse_args()
    args.func(args, parser)
