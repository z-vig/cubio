from __future__ import annotations

# Built-Ins
from typing import TypedDict, NamedTuple
from typing_extensions import Self

# Dependencies
import xarray as xr
import numpy as np
import dask.array as da

# Local
from cubio.cube_size_tools import CubeSize


class MaskBuilder(TypedDict):
    shape: CubeSize
    xdim_name: str
    ydim_name: str
    zdim_name: str


class CubeDims(NamedTuple):
    vdim: str
    hdim: str
    zdim: str


def split_xarray_cube(
    data_array: xr.DataArray,
    cube_dims: CubeDims = CubeDims("Latitude", "Longitude", "Wavelengths"),
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Splits an xarray that represents a data cube into a spatial array and
    a measurement or z array.

    Returns
    -------
    spatial_arr: xr.DataArray
        A 2D xarray DataArray representing the spatial cube dimensions.
    z_arr: xr.DataArray
        A 1D xarray DataArray representing the measurement cube dimension.
    """
    spatial_dims = {
        cube_dims.vdim: slice(None),
        cube_dims.hdim: slice(None),
        cube_dims.zdim: 0,
    }
    z_dim = {
        cube_dims.vdim: 0,
        cube_dims.hdim: 0,
        cube_dims.zdim: slice(None),
    }
    spatial_arr = data_array.isel(spatial_dims)
    z_arr = data_array.isel(z_dim)
    return spatial_arr, z_arr


class CubeMask:
    def __init__(
        self,
        *,
        data_array: xr.DataArray,
        xy_mask: xr.DataArray,
        z_mask: xr.DataArray,
        name: str = "",
    ) -> None:
        self.name = name
        self._spatial_array, self._z_array = split_xarray_cube(data_array)
        self._xymask = xy_mask
        self._zmask = z_mask

    @classmethod
    def transparent(cls, data_array: xr.DataArray) -> Self:
        image_shape = (
            len(data_array.coords["Latitude"]),
            len(data_array.coords["Longitude"]),
        )
        measurement_shape = len(data_array.coords["Wavelengths"])
        spatial, z = split_xarray_cube(data_array)
        xy_mask = spatial.copy(data=da.zeros(shape=image_shape, dtype=np.bool))
        z_mask = z.copy(data=np.zeros(measurement_shape, dtype=bool))

        return cls(
            data_array=data_array,
            xy_mask=xy_mask,
            z_mask=z_mask,
            name="TRANSPARENT",
        )

    def get_xymask(self) -> xr.DataArray:
        return self._xymask

    def set_xymask(self, mask: da.Array | np.ndarray) -> None:
        self._xymask = self._spatial_array.copy(data=mask)

    def get_zmask(self) -> xr.DataArray:
        return self._zmask

    def set_zmask(self, mask: da.Array | np.ndarray) -> None:
        self._zmask = self._z_array.copy(data=mask)

    def add_to_xymask(self, new_mask: da.Array | np.ndarray) -> None:
        new_mask_xr = self._spatial_array.copy(data=new_mask)
        self._xymask |= new_mask_xr

    def add_to_zmask(self, new_mask: da.Array | np.ndarray) -> None:
        new_mask_xr = self._z_array.copy(data=new_mask)
        self._zmask |= new_mask_xr
