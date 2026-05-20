"""
# CubeData Validation

Validation functions for the CuebData Class.
"""

# Dependencies
import xarray as xr

from cubio.cube_dims import CubeDims


def array_is_set(current_array: xr.DataArray | None) -> xr.DataArray:
    """Validates that the current array is not None."""
    if current_array is None:
        raise ValueError("Array is not set.")
    return current_array


def array_dims_match(array: xr.DataArray, cube_dims: CubeDims) -> bool:
    """Checks if the array dimensions match cube dims."""
    if not all([i in cube_dims.as_list() for i in array.dims]):
        return False
    return True
