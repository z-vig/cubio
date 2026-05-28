"""
Cube reading utilities.
"""

# Built-Ins
from pathlib import Path
from typing import Optional

# Dependencies
import xarray as xr
import numpy as np
import rasterio as rio  # type: ignore
from uuid import uuid4

# Local Imports
from cubio.types import (
    suffix_to_format_map,
    NumpyDType,
    RasterioProfile,
    CubeArrayFormat,
    hdr_integer_to_dtype,
)
from cubio.cube_context.envi_hdr_tools import (
    extract_hdr_wavelengths,
    extract_hdr_desc,
    extract_hdr_bbl,
    extract_hdr_band_names,
    extract_dtype,
)
from cubio.geotools.models import GeotransformModel
from cubio.data.crs_wkt_strings import GeographicCRS
from cubio.cube_size_tools import CubeSize
from cubio.cube_context import CubeContext, CubeDataLoader, ContextBuilder
from cubio.cube_data import CubeData


def read_binary_image_file(
    fp: Path, size: CubeSize, data_type: NumpyDType
) -> xr.DataArray:
    """
    Read binary raster image into numpy memmap and then returns xarray
    Dataarray.
    """
    suff = fp.suffix
    binary_fmt = suffix_to_format_map.get(suff)
    if binary_fmt is not None:
        arr = np.memmap(
            fp, dtype=np.dtype(data_type), shape=size.as_tuple(binary_fmt)
        )
        return xr.DataArray(arr)
    else:
        raise NotImplementedError()


def cube_from_json(
    json_fp: Path | str, apply_bbl: bool = True
) -> tuple[CubeContext, CubeData]:
    """
    Reads the json context and loads the data for an image cube.

    Parameters
    ----------
    json_fp: Path to .json file that can be validated to CubeContext object.
    """
    ctxt: CubeContext = CubeContext.from_json(json_fp)
    cb_loader = CubeDataLoader(ctxt)
    cdat: CubeData = cb_loader.lazy_load_data()

    cdat.zcoords = ctxt.measurement_values
    cdat.update_cube_dims(zdim_name=ctxt.measurement_name)

    if apply_bbl:
        cdat.mask.add_to_zmask(ctxt.get_bbl_mask())

    return ctxt, cdat


def cube_from_envi(
    envi_binary_fp: str | Path,
    name: str,
    measurement_name: str = "Wavelength",
    measurement_unit: str = "nm",
) -> tuple[CubeContext, CubeData]:
    """Reads CubeContext and CubeData from envi file."""
    envi_binary_fp = Path(envi_binary_fp)
    with rio.open(envi_binary_fp, "r") as f:
        prf: RasterioProfile = f.profile

    hdr_fp = Path(envi_binary_fp).with_suffix(".hdr")
    wvls = extract_hdr_wavelengths(hdr_fp)
    desc = extract_hdr_desc(hdr_fp)
    bbl = extract_hdr_bbl(hdr_fp)
    band_names = extract_hdr_band_names(hdr_fp)
    dtype = extract_dtype(hdr_fp)

    if wvls == "Wavelengths not found.":
        wvls = [float(i) for i in range(prf["count"])]
    if bbl == "No BBL Found":
        bbl = [1] * prf["count"]
    if band_names == "Band names not found.":
        band_names = [f"Band{i}" for i in range(prf["count"])]

    interlv_test = prf.get("interleave", None)
    interlv: CubeArrayFormat
    if interlv_test is None or interlv_test.lower() == "band":
        interlv = "BIP"
    elif interlv_test.lower() == "pixel":
        interlv = "BIP"
    elif interlv_test.lower() == "line":
        interlv = "BIL"
    else:
        interlv = interlv_test

    if prf["crs"] is None:
        crs_val = str(GeographicCRS.WGS84)
    else:
        crs_val = str(prf["crs"])

    context_dict: ContextBuilder = {
        "name": name,
        "description": desc,
        "data_filename": Path(envi_binary_fp).stem,
        "nrows": prf["height"],
        "ncols": prf["width"],
        "nbands": prf["count"],
        "crs": crs_val,
        "geotransform": GeotransformModel.fromaffine(prf["transform"]),
        "hdr_off": 0,
        "data_type": hdr_integer_to_dtype[dtype],
        "interleave": interlv,
        "nodata": -999,
        "band_names": band_names,
        "measurement_name": measurement_name,
        "measurement_units": measurement_unit,
        "measurement_values": wvls,
        "bad_bands": bbl,
        "id": uuid4(),
    }
    ctxt = CubeContext.from_builder(context_dict)
    ctxt.retrieval_path = Path(envi_binary_fp)

    cb_loader = CubeDataLoader(ctxt)
    cb = cb_loader.lazy_load_data()

    return ctxt, cb


def cube_from_gtif(
    geotiff_fp: str | Path,
    name: str,
    desc: str,
    measurement_name: str = "Measurement",
    measurement_unit: str = "na",
    band_names: Optional[list[str]] = None,
    measurement_vals: Optional[list[float]] = None,
    bbl: Optional[list[int]] = None,
) -> tuple[CubeContext, CubeData]:
    """Reads CubeContext and Cubedata from geotiff."""
    geotiff_fp = Path(geotiff_fp)

    with rio.open(geotiff_fp) as f:
        prf: RasterioProfile = f.profile

    if measurement_vals is None:
        measurement_vals = [float(i) for i in range(prf["count"])]
    if bbl is None:
        bbl = [1] * prf["count"]
    if band_names is None:
        band_names = [f"Band{i}" for i in range(prf["count"])]

    interlv_test = prf.get("interleave", None)
    interlv: CubeArrayFormat
    if interlv_test is None or interlv_test.lower() == "band":
        interlv = "BIP"
    elif interlv_test.lower() == "pixel":
        interlv = "BIP"
    elif interlv_test.lower() == "line":
        interlv = "BIL"
    else:
        interlv = interlv_test

    context_dict: ContextBuilder = {
        "name": name,
        "description": desc,
        "data_filename": Path(geotiff_fp).stem,
        "nrows": prf["height"],
        "ncols": prf["width"],
        "nbands": prf["count"],
        "crs": str(prf["crs"]),
        "geotransform": GeotransformModel.fromaffine(prf["transform"]),
        "hdr_off": 0,
        "data_type": NumpyDType.FLOAT32,
        "interleave": interlv,
        "nodata": -999,
        "band_names": band_names,
        "measurement_name": measurement_name,
        "measurement_units": measurement_unit,
        "measurement_values": measurement_vals,
        "bad_bands": bbl,
        "id": uuid4(),
    }

    ctxt = CubeContext.from_builder(context_dict)
    ctxt.retrieval_path = geotiff_fp
    ctxt.write_json(geotiff_fp.with_suffix(".json"))

    cb_loader = CubeDataLoader(ctxt)
    cb = cb_loader.lazy_load_data()

    return ctxt, cb
