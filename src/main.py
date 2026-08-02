from ssbmv.pipeline.pipeline import VisionPipeline
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import VideoSource
import argparse
from ssbmv.logger import configure_logging
from pathlib import Path
import logging
from datetime import datetime

_logger = logging.getLogger(__name__)

def _get_output_filename(stage: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"ssbmv_{stage}_{timestamp}.json"
        return filename

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to video file to analyze.")
    parser.add_argument("--output", "-o", required=True, type=str, help="Directory to write results to.")
    parser.add_argument("--stage", "-s", required=True, help="name of stage.", choices=["final_destination", "temple", "corneria", "venom"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(args.debug)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
            output_dir = (project_root / output_dir).resolve()

    sprite_path = project_root / 'data' / 'templates'

    if not sprite_path.exists():
        raise RuntimeError(f"Invalid sprite data root path {sprite_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if not output_dir.is_dir():
         raise RuntimeError(f"Output location must be a valid directory: {output_dir}")
         
    sprite_db = SpriteDatabase()
    sprite_db.init(sprite_path)

    pipeline = VisionPipeline(sprite_db, stage=args.stage)
    frame_source = VideoSource(video_frame_source=str(input_path))

    filename = _get_output_filename()
    output_file = output_dir / filename
    pipeline.process(video_source=frame_source, output_file_path=output_file, debug=args.debug)