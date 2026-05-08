"""
# CubeData Validation

Validation functions for the CuebData Class.
"""

# Dependencies
import xarray as xr


def array_is_set(current_array: xr.DataArray | None) -> xr.DataArray:
    """Validates that the current array is not None."""
    if current_array is None:
        raise ValueError("Array is not set.")
    return current_array
