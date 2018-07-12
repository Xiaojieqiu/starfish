from typing import List, Tuple, Union

import pandas as pd
import numpy as np


from starfish.codebook import Codebook
from starfish.image import ImageStack
from starfish.intensity_table import IntensityTable
from ._base import PixelFinderAlgorithmBase


class PixelSpotDetector(PixelFinderAlgorithmBase):

    def __init__(
            self, codebook: Union[str, Codebook], distance_threshold: float=0.5176,
            magnitude_threshold: int=1, area_threshold: int=2,
            crop_size: Tuple[int, int, int]=(0, 40, 40), **kwargs) -> None:
        """Decode an image by first coding each pixel, then combining the results into spots

        Parameters
        ----------
        codebook : str
            file mapping codewords to the genes they represent
        distance_threshold : float
            spots whose codewords are more than this distance from an expected code are filtered (default  0.5176)
        magnitude_threshold : int
            spots with intensity less than this value are filtered (default 1)
        area_threshold : int
            spots with total area less than this value are filtered
        crop_size : Tuple[int, int, int]
            number of pixels to clip around the border of the image, default = (0, 0, 0)

        """
        if isinstance(codebook, str):
            self.codebook = pd.read_json(codebook)
        else:
            self.codebook = codebook
        self.distance_threshold = distance_threshold
        self.magnitude_threshold = magnitude_threshold
        self.area_threshold = area_threshold
        self.crop_size = crop_size

    def find(self, stack: ImageStack) \
            -> Tuple[IntensityTable, List, np.ndarray, np.ndarray, np.ndarray]:
        pixel_intensities = stack.to_pixel_intensities(crop=self.crop_size)
        decoded_intensities = self.codebook.decode_euclidean(
            pixel_intensities,
            max_distance=self.distance_threshold,
            min_intensity=self.magnitude_threshold
        )
        return decoded_intensities.combine_adjacent_features(
            area_threshold=self.area_threshold,
            assume_contiguous=True
        )

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--codebook", help="csv file containing a codebook")
        group_parser.add_argument(
            "--distance-threshold", default=0.5176,
            help="maximum distance a pixel may be from a codeword before it is filtered")
        group_parser.add_argument("--magnitude-threshold", type=float, default=1, help="minimum magnitude of a feature")
        group_parser.add_argument("--area-threshold", type=float, default=2, help="minimum area of a feature")
        # TODO ambrosejcarr: figure out help.
        group_parser.add_argument("--crop-size", type=int, default=40, help="???")
