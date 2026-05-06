import numpy as np
import xarray as xr

from cubio.cube_data import CubeData
from cubio.cube_context import CubeContext
from cubio.types import NumpyDType
from cubio.geotools.models import GeotransformModel


def cube_from_numpy(
    arr: np.ndarray,
    name: str,
    crs: str,
    gtrans: GeotransformModel,
    nodata: float = -999.0,
    measvals: list[float] | None = None,
    bandlbls: list[str] | None = None,
) -> tuple[CubeContext, CubeData]:
    if measvals is None:
        measvals = [float(i) for i in np.arange(arr.shape[2])]
    if bandlbls is None:
        bandlbls = [f"Band {n}" for n in np.arange(arr.shape[2])]
    cc = CubeContext.from_builder(
        {
            "data_filename": name,
            "name": name,
            "description": "From numpy array.",
            "ncols": arr.shape[1],
            "nrows": arr.shape[0],
            "nbands": arr.shape[2],
            "data_type": NumpyDType(str(arr.dtype)),
            "crs": crs,
            "geotransform": gtrans,
            "nodata": nodata,
            "measurement_values": measvals,
            "band_names": bandlbls,
            "measurement_units": "nm",
        }
    )
    cd = CubeData(cc.name, "BIP")
    lons, lats = gtrans.generate_coords(
        height=arr.shape[0], width=arr.shape[1]
    )

    cd.array = xr.DataArray(
        arr,
        coords={
            "latitude": lats,
            "longitude": lons,
            "wavelength": measvals,
        },
        dims=["latitude", "longitude", "wavelength"],
    )

    return cc, cd
