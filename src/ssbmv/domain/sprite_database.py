from enum import Enum
from cv2.typing import MatLike
from cv2 import imread
from functools import lru_cache
from pathlib import Path
import logging
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

class Character(Enum):
    MARIO = 1
    KIRBY = 2

CHARACTER_NAMES: dict[str, Character] = {
   'mario':Character.MARIO,
   'kirby': Character.KIRBY,
}

@dataclass(slots=True)
class SpriteSheet:
    sprite_names: list[str] = field(default_factory=list)
    sprite_img: list[MatLike] = field(default_factory=list)


class SpriteDatabase:
    def __init__(self, sprite_sheet_root_path: Path):
        self.character_sprite_db: dict[Character, SpriteSheet] = {}
        self.sprite_sheet_root_path = sprite_sheet_root_path
    
    def init(self):
        # Init db dict
        for k, v in CHARACTER_NAMES.items():
            self.character_sprite_db[v] = SpriteSheet()
            
        # Load all character sprite sheet data into memory
        _logger.info(f"Loading assets from {self.sprite_sheet_root_path}")
        files = [str(p) for p in self.sprite_sheet_root_path.rglob("*") if p.is_file()]
        for f in files:
            # File names should be in the form {name}_{animation}.{ext}
            parts = f.split(".")[0].split("_")
            if len(parts) <= 1:
                _logger.error("Invalid file name: %s", f)
                continue
            name = parts[0]
            anim = parts[1]
            
            character = CHARACTER_NAMES.get(name.lower(), None)
            if character is None:
                _logger.error("Invalid character name: %s", character)
                continue
            character_sprites = self.character_sprite_db.get(character, None)
            if character_sprites is None:
                _logger.error("Unsupported character %s", character)
            
            img = imread(f)
            if img is None:
                _logger.error("Unable to load file %s", f)
            character_sprites.sprite_names.append(anim)
            character_sprites.sprite_img.append(img)
            
    @lru_cache(maxsize=4)
    def get_sprites_by_character(self, character: Character) -> list[SpriteSheet] | None:
        sprites = self.character_sprite_db.get(character, None)
        return sprites
    
    @lru_cache(maxsize=4)
    def get_palette_by_character(self, character: Character) -> MatLike:
        pass