from enum import Enum
from cv2.typing import MatLike
from cv2 import imread
from functools import lru_cache
from pathlib import Path
import logging
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

class Character(Enum):
    MARIO = 1
    KIRBY = 2

CHARACTER_NAMES: dict[Character, str]={
    Character.MARIO: 'mario',
    Character.KIRBY: 'kirby'
}

@dataclass(slots=True)
class SpriteSheet:
    sprite_names: list[str] = field(default_factory=list)
    sprite_imgs: list[MatLike] = field(default_factory=list)


class SpriteDatabase:
    def __init__(self):
        self.character_sprite_db: dict[str, SpriteSheet] = {}

    def init(self, sprite_sheet_root_path: Path):
        self.character_sprite_db = self._load_character_spritesheets(sprite_sheet_root_path)

    def _load_character_spritesheets(self, root_path: str | Path) -> dict[str, SpriteSheet]:
        """Iterates over a root directory structured as `{character}/{animation}/{num}.jpg`
        and loads each character's sprites into an in-memory SpriteSheet dictionary.
        """
        root = Path(root_path)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid root path: {root_path}")

        character_db: dict[str, SpriteSheet] = {}
        def natural_sort_key(file_path: Path):
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(r"(\d+)", file_path.name)
            ]

        # Supported image extensions
        valid_extensions = {".jpg", ".jpeg", ".png"}

        # Iterate through character folders
        for char_dir in sorted(root.iterdir()):
            if not char_dir.is_dir():
                continue

            char_name = char_dir.name
            sprite_sheet = SpriteSheet()

            # Iterate through animation folders
            for anim_dir in sorted(char_dir.iterdir()):
                if not anim_dir.is_dir():
                    continue

                anim_name = anim_dir.name

                # Collect and naturally sort image files
                image_files = [
                    f for f in anim_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in valid_extensions
                ]
                image_files.sort(key=natural_sort_key)

                # Load each image and append to the character's SpriteSheet
                for img_path in image_files:
                    img = imread(str(img_path))
                    if img is None:
                        continue 

                    sprite_identifier = f"{anim_name}"

                    sprite_sheet.sprite_names.append(sprite_identifier)
                    sprite_sheet.sprite_imgs.append(img)

            character_db[char_name] = sprite_sheet

        return character_db
            
    @lru_cache(maxsize=4)
    def get_sprites_by_character(self, character: Character) -> list[SpriteSheet] | None:
        sprites = self.character_sprite_db.get(character, None)
        return sprites
    
    @lru_cache(maxsize=4)
    def get_palette_by_character(self, character: Character) -> MatLike:
        pass