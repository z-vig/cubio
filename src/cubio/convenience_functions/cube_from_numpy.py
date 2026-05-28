from typing import TypeAlias, Union, TypedDict, Optional

import numpy as np
import xarray as xr
import dask.array as da

from cubio.cube_data import CubeData
from cubio.cube_context import CubeContext
from cubio.types import NumpyDType, CubeArrayFormat, FORMAT_INDICES
from cubio.geotools.models import GeotransformModel

SupportedArray: TypeAlias = Union[np.ndarray, da.Array, xr.DataArray]


class ShapeDict(TypedDict):
    ncols: int
    nrows: int
    nbands: int


def _validate_supported_array(
    array: SupportedArray, format: CubeArrayFormat
) -> xr.DataArray:
    idx = FORMAT_INDICES[format]
    crds = {
        "Ydim": np.arange(array.shape[idx.row]),
        "Xdim": np.arange(array.shape[idx.col]),
        "Zdim": np.arange(array.shape[idx.band]),
    }
    dims = idx.get_dim_names()
    if isinstance(array, np.ndarray) or isinstance(array, da.Array):
        return xr.DataArray(array, coords=crds, dims=dims)
    else:
        return array


def build_cube_context(
    name: str,
    shape_dict: ShapeDict,
    dtype: np.dtype,
    crs: str,
    gtrans: GeotransformModel,
    nodata: float,
    measvals: list[float],
    bandlbls: list[str],
) -> CubeContext:
    cc = CubeContext.from_builder(
        {
            "data_filename": name,
            "name": name,
            "description": "From numpy array.",
            **shape_dict,
            "data_type": NumpyDType(str(dtype)),
            "crs": crs,
            "geotransform": gtrans,
            "nodata": nodata,
            "measurement_values": measvals,
            "band_names": bandlbls,
            "measurement_units": "nm",
        }
    )
    return cc


def cube_from_numpy(
    array: SupportedArray,
    format: CubeArrayFormat,
    cube_context: Optional[CubeContext] = None,
    *,
    name: Optional[str] = None,
    crs: Optional[str] = None,
    gtrans: Optional[GeotransformModel] = None,
    nodata: float = -999.0,
    measvals: Optional[list[float]] = None,
    bandlbls: Optional[list[str]] = None,
) -> tuple[CubeContext, CubeData]:
    arr = _validate_supported_array(array, format)
    idx = FORMAT_INDICES[format]

    if cube_context is not None:
        name = cube_context.name
        crs = cube_context.crs
        gtrans = cube_context.geotransform
        nodata = cube_context.nodata
        measvals = cube_context.measurement_values
        bandlbls = cube_context.band_names
    else:
        if measvals is None:
            measvals = [float(i) for i in np.arange(arr.shape[idx.band])]
        if bandlbls is None:
            bandlbls = [f"Band {n}" for n in np.arange(arr.shape[idx.band])]
        values = {
            "name": name,
            "crs": crs,
            "gtrans": gtrans,
            "nodata": nodata,
        }
        missing = [name for name, value in values.items() if value is None]
        if len(missing) > 0:
            raise ValueError(
                "Missing required aruments: " + ", ".join(missing)
            )
        assert name is not None
        assert crs is not None
        assert gtrans is not None
        assert nodata is not None

    shape_dict: ShapeDict = {
        "nbands": arr.shape[idx.band],
        "ncols": arr.shape[idx.col],
        "nrows": arr.shape[idx.row],
    }

    cc = build_cube_context(
        name, shape_dict, arr.dtype, crs, gtrans, nodata, measvals, bandlbls
    )
    cd = CubeData(cc.name, format)
    cd.array = arr
    cd.geotransform = gtrans

    return cc, cd


if __name__ == "__main__":
    from cubio import cube_from_json

    cc, cd = cube_from_json(
        "D:/moon_data/m3/Gruithuisen_Region/M3T_GDOMES_MOSAIC/M3T_GRUIT_RFL.json"
    )

    test = np.ones((100, 100, 20))

    newcc, newcd = cube_from_numpy(
        test,
        "BIP",
        name="test",
        crs=cc.crs,
        gtrans=cc.geotransform,
        nodata=cc.nodata,
    )

    print(newcc.band_names)
