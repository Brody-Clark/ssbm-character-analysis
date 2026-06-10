from abc import abstractmethod

class FrameSourceBase():
    
    @abstractmethod
    def __iter__(self):
        pass

class VideoSource():
    pass

class ImageSource():
    pass

