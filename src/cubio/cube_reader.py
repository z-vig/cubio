# Built-Ins
from pathlib import Path

# Dependencies
import xarray as xr
import numpy as np

# Local Imports
from cubio.types import suffix_to_format_map, NumpyDType
from cubio.cube_size_tools import CubeSize
from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData


def read_binary_image_file(
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


def read_cube_data(
    json_fp: Path | str, apply_bbl: bool = False
) -> tuple[CubeContext, CubeData]:
    """
    Reads the json context and loads the data for an image cube.

    Parameters
    ----------
    json_fp: Path to .json file that can be validated to CubeContext object.
    """
    ctxt: CubeContext = CubeContext.from_json(json_fp)
    cdat: CubeData = ctxt.lazy_load_data()
    if apply_bbl:
        cdat.mask.add_to_zmask(ctxt.bbl_mask)
    return ctxt, cdat
