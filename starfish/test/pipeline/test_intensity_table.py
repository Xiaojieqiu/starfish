from typing import Tuple

import numpy as np
import pandas as pd

from starfish.constants import Indices, Features
from starfish.intensity_table import IntensityTable
from starfish.codebook import Codebook
from starfish.image import ImageStack
# don't inspect pytest fixtures in pycharm
# noinspection PyUnresolvedReferences
from starfish.test.dataset_fixtures import (
    loaded_codebook, simple_codebook_json, simple_codebook_array, single_synthetic_spot)


def test_empty_intensity_table():
    x = [1, 2]
    y = [2, 3]
    z = [1, 1]
    r = [1, 1]
    spot_attributes = pd.MultiIndex.from_arrays(
        [x, y, z, r],
        names=(Features.X, Features.Y, Features.Z, Features.SPOT_RADIUS))
    image_shape = (2, 4, 3)
    empty = IntensityTable.empty_intensity_table(spot_attributes, 2, 2, image_shape)
    assert empty.shape == (2, 2, 2)
    assert np.sum(empty.values) == 0


def test_synthetic_intensities_generates_correct_number_of_features(loaded_codebook):
    n_spots = 2
    intensities = IntensityTable.synthetic_intensities(loaded_codebook, n_spots=n_spots)
    assert isinstance(intensities, IntensityTable)

    # shape should have n_spots and channels and hybridization rounds equal to the codebook's shape
    assert intensities.shape == (n_spots, *loaded_codebook.shape[1:])


def test_synthetic_intensities_have_correct_number_of_on_features(loaded_codebook):
    n_spots = 2
    intensities = IntensityTable.synthetic_intensities(loaded_codebook, n_spots=n_spots)
    on_features = np.sum(intensities.values != 0)
    # this asserts that the number of features "on" in intensities should be equal to the
    # number of "on" features in the codewords, times the total number of spots in intensities.
    num_on = loaded_codebook.sum((Indices.CH.value, Indices.ROUND.value)).values.mean() * n_spots
    assert on_features == num_on


def feature_data() -> Tuple[Codebook, ImageStack]:
    # This codebook has two codes: on/off and on/on
    # This array has 3 spots: one on/off, one on/on, and one off/on
    # They exist in the first and second z-slice, but not the third.
    code_array = [
        {
            # on/off
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
            ],
            Features.TARGET: "gene_1"
        },
        {
            # on/on
            Features.CODEWORD: [
                {Indices.ROUND.value: 0, Indices.CH.value: 0, Features.CODE_VALUE: 1},
                {Indices.ROUND.value: 1, Indices.CH.value: 0, Features.CODE_VALUE: 1},
            ],
            Features.TARGET: "gene_2"
        }
    ]
    codebook = Codebook.from_code_array(code_array)

    data = np.array(
        [[[[1, 1, 0, 1],  # hyb 0
           [1, 1, 0, 1],
           [0, 0, 0, 0]],

          [[1, 1, 0, 1],
           [1, 1, 0, 1],
           [0, 0, 0, 0]],

          [[0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 0, 0, 0]]],

         [[[1, 1, 0, 0],  # hyb 1
           [1, 1, 0, 0],
           [0, 0, 0, 1]],

          [[1, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 0, 0, 1]],

          [[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]]]]
    )
    data = data.reshape(2, 1, 3, 3, 4)
    image = ImageStack.from_numpy_array(data)
    return codebook, image


def test_combine_adjacent_features():
    codebook, image = feature_data()
    new_intensities = image.to_pixel_intensities()
    # TODO this filtering is causing spots to be discarded, instead of not decoded. Desirable?
    new_intensities = codebook.decode_euclidean(new_intensities, max_distance=0.5, min_intensity=0.5)
    combined_intensities = new_intensities.combine_adjacent_features(area_threshold=0)[0]
    # this is "working", with the caveat that the z-coord is a bit weird and potentially wrong.
    assert np.array_equal(combined_intensities.shape, (2, 1, 1))
    assert np.array_equal(combined_intensities.gene_name, ['gene_2', 'gene_1'])
