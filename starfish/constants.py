from enum import Enum


class AugmentedEnum(Enum):
    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, type(self)) or isinstance(other, str):
            return self.value == other
        return False


class Coordinates(AugmentedEnum):
    Z = 'z'
    Y = 'y'
    X = 'x'


class Indices(AugmentedEnum):
    ROUND = 'h'  # TODO ambrosejcarr: change this to 'r'
    CH = 'c'
    Z = 'z'


class Features:
    """
    contains constants relating to the codebook and feature (spot/pixel) representations of the
    image data
    """

    AXIS = 'features'
    TARGET = 'target'
    CODEWORD = 'codeword'
    CODE_VALUE = 'v'
    SPOT_RADIUS = 'radius'
    SPOT_QUALITY = 'quality'
    Z = 'z'
    Y = 'y'
    X = 'x'
