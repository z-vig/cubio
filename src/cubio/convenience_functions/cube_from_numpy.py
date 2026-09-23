from typing import Literal, TypeAlias, TypedDict

import dask.array as da
import numpy as np
import xarray as xr

from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData
from cubio.geotools.models import GeotransformModel
from cubio.types import FORMAT_INDICES, CubeArrayFormat, NumpyDType

SupportedArray: TypeAlias = np.ndarray | da.Array | xr.DataArray


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
    if isinstance(array, (np.ndarray, da.Array)):
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
    bbl: list[int],
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
            "bad_bands": bbl,
        }
    )
    return cc


def cube_from_numpy(
    array: SupportedArray,
    format: CubeArrayFormat,
    cube_context: CubeContext | None = None,
    *,
    name: str | None = None,
    crs: str | None = None,
    gtrans: GeotransformModel | None = None,
    nodata: float = -999.0,
    measvals: list[float] | Literal["default"] | None = None,
    bandlbls: list[str] | Literal["default"] | None = None,
    bbl: list[int] | Literal["default"] | None = None,
) -> tuple[CubeContext, CubeData]:
    arr = _validate_supported_array(array, format)
    idx = FORMAT_INDICES[format]

    if cube_context is not None:
        if name is None:
            name = cube_context.name
        if crs is None:
            crs = cube_context.crs
        if gtrans is None:
            gtrans = cube_context.geotransform
        if nodata is None:
            nodata = cube_context.nodata

        if measvals is None:
            measvals = cube_context.measurement_values
        elif measvals == "default":
            measvals = [float(i) for i in np.arange(arr.shape[idx.band])]

        if bandlbls is None:
            bandlbls = cube_context.band_names
        elif bandlbls == "default":
            bandlbls = [f"Band {n}" for n in np.arange(arr.shape[idx.band])]
        if bbl is None:
            bbl = cube_context.bad_bands
        elif bbl == "default":
            bbl = [1] * len(measvals)
    else:
        if (measvals is None) or (measvals == "default"):
            measvals = [float(i) for i in np.arange(arr.shape[idx.band])]
        if (bandlbls is None) or (bandlbls == "default"):
            bandlbls = [f"Band {n}" for n in np.arange(arr.shape[idx.band])]
        if (bbl is None) or (bbl == "default"):
            bbl = [1] * len(measvals)
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
        name,
        shape_dict,
        arr.dtype,
        crs,
        gtrans,
        nodata,
        measvals,
        bandlbls,
        bbl,
    )
    cc._retrieval_path = "NoRetrieval"
    cd = CubeData(cc.name, format)
    cd.array = arr.assign_coords({cd.cube_dims.zdim: cc.measurement_values})
    cd.geotransform = gtrans

    cc.measurement_name = cd.cube_dims.zdim

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
