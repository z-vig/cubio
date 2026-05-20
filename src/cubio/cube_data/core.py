"""
The core of the CubeData object.
"""

from __future__ import annotations

# Built-Ins
from typing import Optional

# Dependencies
import xarray as xr
import numpy as np

# Package-Level Imports
from cubio.types import (
    LabelLike,
    CubeArrayFormat,
    FORMAT_INDICES,
)
from cubio.geotools.models import GeotransformModel
from cubio.cube_size_tools import get_cube_size, CubeSize
from cubio.cube_dims import CubeDims

# SubPackage-Level Imports
from .validation import array_is_set, array_dims_match


class CubeDataCore:
    """
    # CubeDataCore
    Core CubeData class. Built for storing the data and metadata of an image
    cube.

    Parameters
    ----------
    name: str
        Name of the cube.
    format: CubeArrayFormat
        Format of the cube array, one of ["BSQ", "BIL", "BIP"].
    x_labels: Optional[LabelLike]
        Optional labels for the x dimension. If not provided, integer
        indexing will be used.
    y_labels: Optional[LabelLike]
        Optional labels for the y dimension. If not provided, integer
        indexing will be used.
    z_labels: Optional[LabelLike]
        Optional labels for the z dimension. If not provided, integer
        indexing will be used.
    x_name: str
        Name of the x dimension. Default is "XAxis".
    y_name: str
        Name of the y dimension. Default is "YAxis".
    z_name: str
        Name of the z dimension. Default is "ZAxis".
    geotransform: Optional[GeotransformModel]
        Optional geotransform for the cube. If provided, it will be used to
        generate the x and y coordinate arrays, and any provided x and y
        labels will be overwritten.
    nodata: float | int
        Value to use for nodata. Default is -999.
    """

    def __init__(
        self,
        name: str,
        format: CubeArrayFormat,
        *,
        cube_dims: CubeDims = CubeDims.default(),
        geotransform: Optional[GeotransformModel] = None,
        crs: Optional[str] = None,
        nodata: float | int = -999,
    ) -> None:
        self.name: str = name  # Name of the Cube
        self._gtrans = geotransform  # Geotransform, if there is one.
        self._crs = crs  # CRS, if there is one.
        self.nodata = nodata  # No data value, default = -999

        # Setting the format to one of {"BIL", "BIP", "BSQ"}
        self._fmt: CubeArrayFormat = format
        self.fmt = self._fmt

        self._array: xr.DataArray | None = None
        self._xcoords: Optional[LabelLike] = None
        self._ycoords: Optional[LabelLike] = None
        self._zcoords: Optional[LabelLike] = None

        self.cube_dims = cube_dims
        self._shape: CubeSize | None = None

    @property
    def fmt(self) -> CubeArrayFormat:
        return self._fmt

    @fmt.setter
    def fmt(self, value: CubeArrayFormat) -> None:
        self._fmt = value
        idx = FORMAT_INDICES[self._fmt]
        self.rowindex = idx.row
        self.colindex = idx.col
        self.bandindex = idx.band

    @property
    def shape(self) -> CubeSize:
        if self._shape is None:
            self._array = array_is_set(self._array)
            self._shape = get_cube_size(self._array, self.fmt)
        return self._shape

    @property
    def xcoords(self) -> LabelLike:
        if self._xcoords is None:
            raise RuntimeError("X Coords not set.")
        return self._xcoords

    @xcoords.setter
    def xcoords(self, xcoords: LabelLike) -> None:
        self._array = array_is_set(self._array)
        update_len = len(xcoords)
        current_len = self._array.sizes[self.cube_dims.hdim]
        if update_len != current_len:
            raise ValueError(
                f"Length of xcoords: ({update_len}) does not "
                f"match current data array size: ({current_len})"
            )
        self._array = self._array.assign_coords({self.cube_dims.hdim: xcoords})
        self._xcoords = xcoords

    @property
    def ycoords(self) -> LabelLike:
        if self._ycoords is None:
            raise RuntimeError("Y Coords not set.")
        return self._ycoords

    @ycoords.setter
    def ycoords(self, ycoords: LabelLike) -> None:
        self._array = array_is_set(self._array)
        update_len = len(ycoords)
        current_len = self._array.sizes[self.cube_dims.vdim]
        if update_len != current_len:
            raise ValueError(
                f"Length of ycoords: ({update_len}) does not "
                f"match current data array size: ({current_len})"
            )
        self._array = self._array.assign_coords({self.cube_dims.vdim: ycoords})
        self._ycoords = ycoords

    @property
    def zcoords(self) -> LabelLike:
        if self._zcoords is None:
            raise RuntimeError("Z Coords not set.")
        return self._zcoords

    @zcoords.setter
    def zcoords(self, zcoords: LabelLike) -> None:
        self._array = array_is_set(self._array)
        update_len = len(zcoords)
        current_len = self._array.sizes[self.cube_dims.zdim]
        if update_len != current_len:
            raise ValueError(
                f"Length of zcoords: ({update_len}) does not "
                f"match current data array size: ({current_len})"
            )
        self._array = self._array.assign_coords({self.cube_dims.zdim: zcoords})
        self._zcoords = zcoords

    @property
    def array(self) -> xr.DataArray:
        self._array = array_is_set(self._array)
        self._array.name = "data"
        return self._array

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        if not array_dims_match(value, self.cube_dims):
            raise ValueError(
                f"DataArray has dims: {value.dims} that do not match "
                f"the registered CubeDims: {self.cube_dims.as_list()}"
            )
        if value.ndim == 2:
            print("EXPANDING DIMS")
            value = value.expand_dims(
                dim={self.cube_dims.zdim: 1}, axis=self.bandindex
            )
        self._shape = get_cube_size(value, self.fmt)
        self._array = self.set_array_coords(value)
        self._xcoords = np.array(self._array.coords[self.cube_dims.hdim])
        self._ycoords = np.array(self._array.coords[self.cube_dims.vdim])
        self._zcoords = np.array(self._array.coords[self.cube_dims.zdim])
        self._post_array_setting_config()

    def set_array_coords(self, value: xr.DataArray) -> xr.DataArray:
        value = value.assign_coords(
            {
                self.cube_dims.vdim: np.arange(
                    0, value.sizes[self.cube_dims.vdim]
                ),
                self.cube_dims.hdim: np.arange(
                    0, value.sizes[self.cube_dims.hdim]
                ),
                self.cube_dims.zdim: np.arange(
                    0, value.sizes[self.cube_dims.zdim]
                ),
            }
        )
        return value

    def update_cube_dims(
        self,
        cube_dims: Optional[CubeDims] = None,
        *,
        vdim_name: Optional[str] = None,
        hdim_name: Optional[str] = None,
        zdim_name: Optional[str] = None,
    ) -> None:
        arr = array_is_set(self._array)
        if cube_dims is None:
            cube_dims = CubeDims(
                vdim_name or self.cube_dims.vdim,
                hdim_name or self.cube_dims.hdim,
                zdim_name or self.cube_dims.zdim,
            )

        arr = arr.rename(
            {
                self.cube_dims.vdim: cube_dims.vdim,
                self.cube_dims.hdim: cube_dims.hdim,
                self.cube_dims.zdim: cube_dims.zdim,
            }
        )
        self.cube_dims = cube_dims
        self._array = arr

    def _post_array_setting_config(self) -> None:
        return None
