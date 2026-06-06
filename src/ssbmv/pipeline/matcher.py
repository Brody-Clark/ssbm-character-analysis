from src.ssbmv.domain.sprite_database import SpriteDatabase
import logging

_logger = logging.getLogger(__name__)

class _CharacterMatcher:
    def __init__(self,  sprite_database: SpriteDatabase):
        self.sprite_db = sprite_database
        pass
    
    def match(self):
        pass
    
class _AnimationMatcher:
    def __init__(self, sprite_database: SpriteDatabase):
        self.sprite_db = sprite_database
        pass
    
    def match(self):
        pass
    
class Matcher:
    def __init__(self,  sprite_database: SpriteDatabase):
        self.character_matcher = _CharacterMatcher(sprite_database=sprite_database)
        self.animation_matcher = _AnimationMatcher(sprite_database=sprite_database)