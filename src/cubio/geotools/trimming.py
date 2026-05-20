"""
Trimming utilities for xarray DataArrays.
"""

from dataclasses import dataclass

import xarray as xr

from cubio.cube_mask import CubeDims
from cubio.cube_data import CubeData
from cubio.geotools.models import GeotransformModel


@dataclass
class TrimResult:
    array: xr.DataArray
    gtrans: GeotransformModel
    vindex: slice
    hindex: slice


def trim_nan_borders(data: xr.DataArray, cube_dims: CubeDims) -> TrimResult:
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

    vslice = slice(first_valid_row, last_valid_row)
    hslice = slice(first_valid_col, last_valid_col)
    data = data.isel(
        {
            cube_dims.vdim: vslice,
            cube_dims.hdim: hslice,
            cube_dims.zdim: slice(None),
        }
    )

    upper_left_x = data.coords[cube_dims.hdim][0]
    upper_left_y = data.coords[cube_dims.vdim][0]
    lower_right_x = data.coords[cube_dims.hdim][-1]
    lower_right_y = data.coords[cube_dims.vdim][-1]

    new_gtrans = GeotransformModel.fromarraysize(
        upper_left_y,
        upper_left_x,
        lower_right_y,
        lower_right_x,
        data.sizes[cube_dims.vdim],
        data.sizes[cube_dims.hdim],
    )

    return TrimResult(data, new_gtrans, vslice, hslice)


def trim_cubedata(data: CubeData) -> None:
    trim = trim_nan_borders(data.masked_array, data.cube_dims)
    sel_dict_spatial = {
        data.cube_dims.vdim: trim.vindex,
        data.cube_dims.hdim: trim.hindex,
    }
    sel_dict_z = {data.cube_dims.zdim: slice(None)}
    data.array = data.array.isel({**sel_dict_spatial, **sel_dict_z})
    data.geotransform = trim.gtrans
    old_xymask = data.mask.get_xymask().isel(sel_dict_spatial)
    old_zmask = data.mask.get_zmask()
    data.reset_mask()
    old_xymask = old_xymask.assign_coords(
        {
            data.cube_dims.vdim: data.array.coords[data.cube_dims.vdim],
            data.cube_dims.hdim: data.array.coords[data.cube_dims.hdim],
        }
    )
    data.mask.add_to_xymask(old_xymask)
    data.mask.add_to_zmask(old_zmask)
