# Dependencies
import xarray as xr


def array_is_set(current_array: xr.DataArray | None) -> xr.DataArray:
    if current_array is None:
        raise ValueError("Array is not set.")
    return current_array
