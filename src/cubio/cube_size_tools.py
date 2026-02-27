"""
#### `cube_size_tools`
Tools for dealing with the shape/size interpretation of an image cube array.
"""

# Built-ins
from dataclasses import dataclass
from typing import Literal

# Dependencies
import numpy as np
import xarray as xr

# Local imports
from cubio.types import CubeArrayFormat


@dataclass
class CubeSize:
    nrows: int
    ncolumns: int
    nbands: int

    def as_tuple(self, interleave: CubeArrayFormat) -> tuple[int, int, int]:
        if interleave == "BIL":
            return (self.nrows, self.nbands, self.ncolumns)
        elif interleave == "BIP":
            return (self.nrows, self.ncolumns, self.nbands)
        elif interleave == "BSQ":
            return (self.nbands, self.ncolumns, self.nrows)


def get_cube_size(
    arr: np.ndarray | np.memmap | xr.DataArray, format: CubeArrayFormat
) -> CubeSize:
    if format == "BIL":
        lines, bands, samples = arr.shape
    elif format == "BIP":
        lines, samples, bands = arr.shape
    elif format == "BSQ":
        bands, samples, lines = arr.shape

    return CubeSize(lines, samples, bands)


def transpose_cube(
    src: CubeArrayFormat,
    dst: CubeArrayFormat | Literal["RASTERIO"],
    arr: xr.DataArray,
):
    if src == "BIL":
        if dst == "BIL":
            return arr
        elif dst == "BIP":
            return arr.transpose(*(arr.dims[0], arr.dims[2], arr.dims[1]))
        elif dst == "BSQ":
            return arr.transpose(*(arr.dims[1], arr.dims[2], arr.dims[0]))
        elif dst == "RASTERIO":
            return arr.transpose(*(arr.dims[1], arr.dims[0], arr.dims[2]))
    elif src == "BIP":
        if dst == "BIL":
            return arr.transpose(*(arr.dims[0], arr.dims[2], arr.dims[1]))
        elif dst == "BIP":
            return arr
        elif dst == "BSQ":
            return arr.transpose(*(arr.dims[2], arr.dims[1], arr.dims[0]))
        elif dst == "RASTERIO":
            return arr.transpose(*(arr.dims[2], arr.dims[0], arr.dims[1]))
    elif src == "BSQ":
        if dst == "BIL":
            return arr.transpose(*(arr.dims[2], arr.dims[0], arr.dims[1]))
        elif dst == "BIP":
            return arr.transpose(*(arr.dims[2], arr.dims[1], arr.dims[0]))
        elif dst == "BSQ":
            return arr
        elif dst == "RASTERIO":
            return arr.transpose(*(arr.dims[0], arr.dims[2], arr.dims[1]))
