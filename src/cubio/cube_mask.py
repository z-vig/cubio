from __future__ import annotations

# Built-Ins
from typing import TYPE_CHECKING
from typing_extensions import Self

# Dependencies
import xarray as xr
import numpy as np

# Local

if TYPE_CHECKING:
    from .cube_data import CubeData


class CubeMask:
    def __init__(
        self,
        parent: CubeData,
        *,
        xy_mask: xr.DataArray | None = None,
        z_mask: xr.DataArray | None = None,
    ) -> None:
        self.shape = parent.shape
        if (xy_mask is not None) and (xy_mask.dtype != bool):
            raise ValueError(
                "XY Mask must be of dtype bool not" f" {xy_mask.dtype}"
            )
        if (z_mask is not None) and (z_mask.dtype != bool):
            raise ValueError("Z Mask must be of dtype: bool")

        self._xymask = xy_mask
        self._zmask = z_mask
        self.dims = (parent.ydim_name, parent.xdim_name, parent.zdim_name)
        self._parent = parent

    @classmethod
    def transparent(cls, parent: CubeData) -> Self:
        return cls(
            parent,
            xy_mask=xr.DataArray(
                np.zeros(
                    (parent.shape.nrows, parent.shape.ncolumns), dtype=bool
                ),
                dims=(parent.ydim_name, parent.xdim_name),
            ),
            z_mask=xr.DataArray(
                np.zeros(parent.shape.nbands, dtype=bool),
                dims=(parent.zdim_name),
            ),
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
        self._xymask = value
        self._parent.mask = self

    @property
    def zmask(self) -> xr.DataArray:
        if self._zmask is None:
            raise ValueError("XY Mask not set yet.")
        self._zmask.rename(self.dims[2])
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
        self._parent.mask = self

    def add_to_xymask(self, new_mask: xr.DataArray) -> None:
        if new_mask.dtype != bool:
            raise ValueError(f"New Mask is not of type bool, {new_mask.dtype}")
        if self._xymask is None:
            raise ValueError("XYMask not set yet.")
        if new_mask.ndim != 2:
            raise ValueError(
                f"New mask has invalid dimensions: {new_mask.ndim}"
            )
        new_mask.rename((self.dims[0], self.dims[1]))
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
        new_mask.rename((self.dims[2]))
        self.zmask = self._zmask | new_mask
