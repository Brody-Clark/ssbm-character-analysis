from abc import abstractmethod
from ssbmv.domain.models import Frame

class FrameSourceBase():
    
    @abstractmethod
    def __iter__(self) -> Frame:
        pass

class VideoSource(FrameSourceBase):
    pass

class ImageSource(FrameSourceBase):
    pass

