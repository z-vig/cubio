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
)
from cubio.envi_hdr_tools import (
    extract_hdr_wavelengths,
    extract_hdr_desc,
    extract_hdr_bbl,
    extract_hdr_band_names,
)
from cubio.geotools.models import GeotransformModel
from cubio.cube_size_tools import CubeSize
from cubio.cube_context import CubeContext, ContextBuilder
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


def cubedata_from_json_file(
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


def cubedata_from_envi_file(
    envi_binary_fp: str | Path,
    name: str,
    measurement_name: str = "Wavelength",
    measurement_unit: str = "nm",
) -> tuple[CubeContext, CubeData]:
    envi_binary_fp = Path(envi_binary_fp)
    with rio.open(envi_binary_fp, "r") as f:
        prf: RasterioProfile = f.profile

    hdr_fp = Path(envi_binary_fp).with_suffix(".hdr")
    wvls = extract_hdr_wavelengths(hdr_fp)
    desc = extract_hdr_desc(hdr_fp)
    bbl = extract_hdr_bbl(hdr_fp)
    band_names = extract_hdr_band_names(hdr_fp)

    if wvls == "Wavelengths not found.":
        wvls = [float(i) for i in range(prf["count"])]
    if bbl == "No BBL Found":
        bbl = [1] * prf["count"]
    if band_names == "Band names not found.":
        band_names = [f"Band{i}" for i in range(prf["count"])]

    interlv = prf.get("interleave", None)
    if interlv is None:
        raise ValueError("ENVI File does not specify interleave.")

    context_dict: ContextBuilder = {
        "name": name,
        "description": desc,
        "data_filename": Path(Path(envi_binary_fp).stem),
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
        "measurement_values": wvls,
        "bad_bands": bbl,
        "id": uuid4(),
    }

    ctxt = CubeContext.from_builder(context_dict)
    ctxt.write_envi_hdr(hdr_fp)
    cb = ctxt.lazy_load_data()

    return ctxt, cb


def cubedata_from_geotiff(
    geotiff_fp: str | Path,
    name: str,
    desc: str,
    measurement_name: str = "Wavelength",
    measurement_unit: str = "nm",
    band_names: Optional[list[str]] = None,
    measurement_vals: Optional[list[float]] = None,
    bbl: Optional[list[int]] = None,
) -> tuple[CubeContext, CubeData]:
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
    if interlv_test is None or interlv_test == "band":
        interlv = "BIP"
    else:
        interlv = interlv_test

    context_dict: ContextBuilder = {
        "name": name,
        "description": desc,
        "data_filename": Path(Path(geotiff_fp).stem),
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
    ctxt._retrieval_path = geotiff_fp
    ctxt.write_json(geotiff_fp.with_suffix(".json"))

    cb = ctxt.lazy_load_data()

    return ctxt, cb
