from ssbmv.pipeline.pipeline import VisionPipeline
from ssbmv.domain.sprite_database import SpriteDatabase
import argparse
from ssbmv.logger import configure_logging
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("")
    args = parser.parse_args()

    configure_logging(args.debug)
    
    sprite_path = Path.cwd() / 'assets' / 'sprites'
    if not sprite_path.exists():
        raise RuntimeError(f"Invalid sprite data root path {sprite_path}")
    
    sprite_db = SpriteDatabase(sprite_sheet_root_path=sprite_path)
    sprite_db.init()
    pipeline = VisionPipeline(sprite_db)