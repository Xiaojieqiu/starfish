from starfish.image import ImageStack
from starfish.pipeline.algorithmbase import AlgorithmBase


class SpotFinderAlgorithmBase(AlgorithmBase):
    def find(self, hybridization_image: ImageStack):
        """Find spots."""
        raise NotImplementedError()
