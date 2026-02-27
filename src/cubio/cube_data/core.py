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
from cubio.types import LabelLike, CubeArrayFormat, FORMAT_INDICES
from cubio.geotools.models import GeotransformModel
from cubio.cube_size_tools import get_cube_size, CubeSize

# SubPackage-Level Imports
from .validation import array_is_set


class CubeDataCore:
    """
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
        xcoord_label: Optional[LabelLike] = None,
        ycoord_label: Optional[LabelLike] = None,
        zcoord_label: Optional[LabelLike] = None,
        x_name: str = "XAxis",
        y_name: str = "YAxis",
        z_name: str = "ZAxis",
        geotransform: Optional[GeotransformModel] = None,
        nodata: float | int = -999,
    ) -> None:
        self.name: str = name  # Name of the Cube
        self._gtrans = geotransform  # Geotransform, if there is one.
        self.nodata = nodata  # No data value, default = -999

        # Setting the format to one of {"BIL", "BIP", "BSQ"}
        self._fmt: CubeArrayFormat = format
        self.fmt = self._fmt

        self._array: xr.DataArray | None = None

        self._xcoords = xcoord_label
        self._ycoords = ycoord_label
        self._zcoords = zcoord_label

        self.xdim_name = x_name
        self.ydim_name = y_name
        self.zdim_name = z_name

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
            self._xcoords = np.arange(0, self.shape.ncolumns)
        return self._xcoords

    @property
    def ycoords(self) -> LabelLike:
        if self._ycoords is None:
            self._ycoords = np.arange(0, self.shape.ncolumns)
        return self._ycoords

    @property
    def zcoords(self) -> LabelLike:
        if self._zcoords is None:
            self._zcoords = np.arange(0, self.shape.ncolumns)
        return self._zcoords

    @property
    def array(self) -> xr.DataArray:
        self._array = array_is_set(self._array)
        self._array.name = "data"
        return self._array

    @array.setter
    def array(self, value: xr.DataArray) -> None:
        if value.ndim == 2:
            value = value.expand_dims(dim={self.zdim_name: 1}, axis=2)
        self._shape = get_cube_size(value, self.fmt)
        self._array = self._create_labeled_dataarray(value)  # Labeled array.

    def _create_dims_tuple(self) -> tuple[str, str, str]:
        """
        Creates a tuple of string dim names in the correct order for the
        current array format.
        """
        _n = np.array(
            [self.ydim_name, self.xdim_name, self.zdim_name]
        )  # names
        dims = (_n[self.rowindex], _n[self.colindex], _n[self.bandindex])
        return dims

    def _create_coords_dict(self) -> dict[str, LabelLike]:
        # All coordinate arrays are index arrays, if not set.
        coordinate_dict = {
            self.xdim_name: self.xcoords,
            self.ydim_name: self.ycoords,
            self.zdim_name: self.zcoords,
        }
        return coordinate_dict

    def _create_labeled_dataarray(self, value: xr.DataArray) -> xr.DataArray:
        """
        Creates a new xarray dataarray from an existing dataarray but with the
        correct coords and dimension names in the correct order for the current
        cube format.
        """
        dims = self._create_dims_tuple()
        crds = self._create_coords_dict()
        return xr.DataArray(
            value.data,
            coords=crds,
            dims=dims,
        )
