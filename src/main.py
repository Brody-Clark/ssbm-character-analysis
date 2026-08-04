from ssbmv.pipeline.pipeline import VisionPipeline
from ssbmv.domain.sprite_database import SpriteDatabase
from ssbmv.source.frame_source import VideoSource
import argparse
from ssbmv.logger import configure_logging
from pathlib import Path
import logging
from contextlib import ExitStack
import sys

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="Path to video file to analyze."
    )
    parser.add_argument(
        "--output",
        choices=["stdout", "file"],
        default="stdout",
        help="Choose where to send the output (default: stdout)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the output file (required if --output is set to 'file')",
    )
    parser.add_argument(
        "--stage",
        "-s",
        required=True,
        help="name of stage.",
        choices=["final_destination", "temple", "corneria", "venom"],
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging(args.debug)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()
    if not input_path.exists():
        parser.error(f"Input video not found: {input_path}")

    if args.output == "file" and not args.file:
        parser.error("--file is required when --output is set to 'file'.")

    sprite_path = project_root / "data" / "templates"
    if not sprite_path.exists():
        raise RuntimeError(f"Invalid sprite data root path {sprite_path}")

    with ExitStack() as stack:
        # Determine the output stream
        if args.output == "file":
            output_file = Path(args.output)
            if output_file.suffix.lower() != ".json":
                parser.error("The output file must have a .json extension.")
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                stack.enter_context(out_stream=open(args.file, "w", encoding="utf-8"))
            except OSError as e:
                print(f"Error opening file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            out_stream = sys.stdout

        try:
            sprite_db = SpriteDatabase()
            sprite_db.init(sprite_path)
            pipeline = VisionPipeline(sprite_db, stage=args.stage)
            frame_source = VideoSource(video_frame_source=str(input_path))
        except Exception as e:
            print(f"Initialization failed: {e}")
            print(f"Aborting.")
            stack.close()
            sys.exit(1)

        # Process video
        pipeline.process(
            video_source=frame_source, output_stream=out_stream, debug=args.debug
        )
