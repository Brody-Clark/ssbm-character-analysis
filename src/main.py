from ssbmv.pipeline.pipeline import VisionPipeline
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import VideoSource
import argparse
from ssbmv.logger import configure_logging
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to video file to analyze.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(args.debug)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    sprite_path = project_root / 'data' / 'templates'

    if not sprite_path.exists():
        raise RuntimeError(f"Invalid sprite data root path {sprite_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    sprite_db = SpriteDatabase()
    sprite_db.init(sprite_path)

    pipeline = VisionPipeline(sprite_db)
    frame_source = VideoSource(video_frame_source=str(input_path))

    pipeline.process(video_source=frame_source, debug=args.debug)