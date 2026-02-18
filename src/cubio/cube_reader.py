# Built-Ins
from pathlib import Path

# Dependencies
import xarray as xr
import numpy as np

# Local Imports
from cubio.types import suffix_to_format_map, NumpyDType
from cubio.cube_size_tools import CubeSize


def cube_reader(
    fp: Path, size: CubeSize, data_type: NumpyDType
) -> xr.DataArray:
    suff = fp.suffix
    binary_fmt = suffix_to_format_map.get(suff)
    if binary_fmt is not None:
        arr = np.memmap(
            fp, dtype=np.dtype(data_type), shape=size.as_tuple(binary_fmt)
        )
        return xr.DataArray(arr)
    else:
        raise NotImplementedError()
