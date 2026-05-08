"""
Trimming utilities for xarray DataArrays.
"""

import xarray as xr

from cubio.cube_mask import CubeDims
from cubio.cube_data import CubeData


def trim_nan_borders(data: xr.DataArray, cube_dims: CubeDims) -> xr.DataArray:
    # All valid data
    valid_data = data.notnull().any(dim=cube_dims.zdim)
    valid_rows = valid_data.any(dim=cube_dims.hdim)
    valid_cols = valid_data.any(dim=cube_dims.vdim)

    first_valid_row = valid_rows.argmax(dim=cube_dims.vdim).compute().values
    first_valid_col = valid_cols.argmax(dim=cube_dims.hdim).compute().values

    last_valid_row = (
        valid_rows.size
        - valid_rows[::-1].argmax(dim=cube_dims.vdim).compute().values
    )
    last_valid_col = (
        valid_cols.size
        - valid_cols[::-1].argmax(dim=cube_dims.hdim).compute().values
    )

    return data.isel(
        {
            cube_dims.vdim: slice(first_valid_row, last_valid_row),
            cube_dims.hdim: slice(first_valid_col, last_valid_col),
            cube_dims.zdim: slice(None),
        }
    )


def trim_cubedata(data: CubeData, apply_mask: bool = True) -> None:
    pass
