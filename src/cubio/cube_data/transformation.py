from cubio.cube_size_tools import transpose_cube
from cubio.types import CubeArrayFormat

from .core import CubeDataCore


class TransformationMixIn(CubeDataCore):
    """
    ### TransformationMixIn

    Adds data transforms to the CubeData class.
    """

    def transpose_to(self, format: CubeArrayFormat) -> None:
        old_format = self.fmt
        self.fmt = format
        new_arr = transpose_cube(old_format, format, self.array)
        self.array = new_arr
