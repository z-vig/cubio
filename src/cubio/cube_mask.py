from __future__ import annotations

# Built-Ins
from typing import TypedDict
from typing_extensions import Self

# Dependencies
import xarray as xr
import numpy as np

# Local
from cubio.cube_size_tools import CubeSize


class MaskBuilder(TypedDict):
    shape: CubeSize
    xdim_name: str
    ydim_name: str
    zdim_name: str


class CubeMask:
    def __init__(
        self,
        *,
        shape: CubeSize,
        xdim_name: str,
        ydim_name: str,
        zdim_name: str,
        xy_mask: xr.DataArray | None = None,
        z_mask: xr.DataArray | None = None,
    ) -> None:
        self.shape = shape
        if (xy_mask is not None) and (xy_mask.dtype != bool):
            raise ValueError(
                "XY Mask must be of dtype bool not" f" {xy_mask.dtype}"
            )
        if (z_mask is not None) and (z_mask.dtype != bool):
            raise ValueError("Z Mask must be of dtype: bool")

        self._xymask = xy_mask
        self._zmask = z_mask
        self.xdim_name = xdim_name
        self.ydim_name = ydim_name
        self.zdim_name = zdim_name

    @classmethod
    def transparent(
        cls,
        shape: CubeSize,
        xdim_name: str,
        ydim_name: str,
        zdim_name: str,
    ) -> Self:
        xy_mask = xr.DataArray(
            np.zeros((shape.nrows, shape.ncolumns), dtype=bool),
            dims=(ydim_name, xdim_name),
        )
        z_mask = xr.DataArray(
            np.zeros(shape.nbands, dtype=bool),
            dims=(zdim_name),
        )
        return cls(
            shape=shape,
            xdim_name=xdim_name,
            ydim_name=ydim_name,
            zdim_name=zdim_name,
            xy_mask=xy_mask,
            z_mask=z_mask,
        )

    @property
    def xymask(self) -> xr.DataArray:
        if self._xymask is None:
            raise ValueError("XY Mask not set yet.")
        return self._xymask

    @xymask.setter
    def xymask(self, value: xr.DataArray) -> None:
        if value.dtype != bool:
            raise ValueError(f"XY Mask must be of type bool not {value.dtype}")
        if value.ndim != 2:
            raise ValueError(
                f"Invalid number of xy mask dimensions: {value.ndim}"
            )
        if self._xymask is None:
            raise ValueError("XY Mask is not set.")
        self._xymask.rename((self.ydim_name, self.xdim_name))
        self._xymask = value

    @property
    def zmask(self) -> xr.DataArray:
        if self._zmask is None:
            raise ValueError("Z Mask not set yet.")
        self._zmask.rename(self.zdim_name)
        return self._zmask

    @zmask.setter
    def zmask(self, value: xr.DataArray) -> None:
        if value.dtype != bool:
            raise ValueError(f"Z Mask must be of type bool not {value.dtype}")
        if value.ndim != 1:
            raise ValueError(
                f"Invalid number of xy mask dimensions: {value.ndim}"
            )
        self._zmask = value

    def add_to_xymask(self, new_mask: xr.DataArray) -> None:
        if new_mask.dtype != bool:
            raise ValueError(f"New Mask is not of type bool, {new_mask.dtype}")
        if self._xymask is None:
            raise ValueError("XYMask not set yet.")
        if new_mask.ndim != 2:
            raise ValueError(
                f"New mask has invalid dimensions: {new_mask.ndim}"
            )
        new_mask.rename((self.ydim_name, self.xdim_name))
        self.xymask = self._xymask | new_mask

    def add_to_zmask(self, new_mask: xr.DataArray) -> None:
        if new_mask.dtype != bool:
            raise ValueError(f"New Mask is not of type bool, {new_mask.dtype}")
        if self._zmask is None:
            raise ValueError("ZMask not set yet.")
        if new_mask.ndim != 1:
            raise ValueError(
                f"New mask has invalid dimensions: {new_mask.ndim}"
            )
        new_mask.rename((self.zdim_name))
        self.zmask = self._zmask | new_mask
