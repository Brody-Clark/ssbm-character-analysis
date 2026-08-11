"""Load sprite and HUD assets into memory for template matching."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from cv2 import imread
from cv2.typing import MatLike

_logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class SpriteSheet:
    """A collection of sprite images and their animation identifiers."""

    sprite_names: list[str] = field(default_factory=list)
    sprite_imgs: list[MatLike] = field(default_factory=list)


class SpriteDatabase:
    """Load and expose sprite assets used by the matching pipeline."""

    def __init__(self):
        self.character_sprite_db: dict[str, SpriteSheet] = {}
        self.character_hud_db: dict[str, MatLike] = {}

    def init(self, asset_path: Path):
        """Populate the database from a directory tree of sprite and HUD assets."""
        self.character_sprite_db = self._load_character_spritesheets(
            asset_path / "sprites"
        )
        self.character_hud_db = self._load_character_huds(asset_path / "huds")

    def _load_character_huds(self, root_path: str | Path) -> dict[str, MatLike]:
        """Load HUD icon images from a flat directory of image files."""
        root = Path(root_path)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid root path: {root_path}")

        sprite_db: dict[str, MatLike] = {}
        for img_path in sorted(root.iterdir()):
            if (
                not img_path.is_file()
                or img_path.suffix.lower() not in VALID_EXTENSIONS
            ):
                _logger.debug("File %s is not valid. Skipping.", str(img_path))
                continue

            img = imread(str(img_path))
            if img is not None:
                sprite_db[img_path.stem] = img

        return sprite_db

    def _load_character_spritesheets(
        self, root_path: str | Path
    ) -> dict[str, SpriteSheet]:
        """Load character sprite sheets from a directory of per-character animations."""
        root = Path(root_path)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid root path: {root_path}")

        def natural_sort_key(file_path: Path):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(r"(\d+)", file_path.name)
            ]

        character_db: dict[str, SpriteSheet] = {}
        for char_dir in sorted(root.iterdir()):
            if not char_dir.is_dir():
                continue

            sprite_sheet = SpriteSheet()
            for anim_dir in sorted(char_dir.iterdir()):
                if not anim_dir.is_dir():
                    continue

                image_files = [
                    path
                    for path in anim_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
                ]
                image_files.sort(key=natural_sort_key)

                for img_path in image_files:
                    img = imread(str(img_path))
                    if img is None:
                        continue
                    sprite_sheet.sprite_names.append(anim_dir.name)
                    sprite_sheet.sprite_imgs.append(img)

            character_db[char_dir.name] = sprite_sheet

        return character_db
