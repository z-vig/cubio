# Built-Ins
from pathlib import Path
from uuid import uuid4

# Dependencies
import rasterio as rio  # type: ignore

# Local Imports
from cubio.types import RasterioProfile, NumpyDType
from cubio.cube_context import CubeContext, ContextBuilder
from cubio.geotools.models import GeotransformModel
from cubio.data.crs_wkt_strings import GeographicCRS
from cubio.envi_hdr_tools import (
    extract_hdr_wavelengths,
    extract_hdr_bbl,
    extract_hdr_band_names,
    extract_hdr_desc,
)


def read_spectral_envi_file_context(fp: Path | str, name: str) -> CubeContext:
    """
    Reads the context data from an ENVI cube file representing spectral data.

    Parameters
    ----------
    fp : Path | str
        File path to ENVI cube.
    name : str
        Name of the CubeContext object.

    Returns
    -------
    CubeContext
        Relevant context data about the cube.

    Raises
    ------
    ValueError
        If the .hdr file does not contain a wavelength field.
    """

    with rio.open(fp, "r") as f:
        prf: RasterioProfile = f.profile

    hdr_fp = Path(fp).with_suffix(".hdr")
    wvls = extract_hdr_wavelengths(hdr_fp)
    desc = extract_hdr_desc(hdr_fp)
    bbl = extract_hdr_bbl(hdr_fp)
    if wvls == "Wavelengths not found.":
        raise ValueError(
            "File is not a spectral envi file. Try"
            "`read_measurement_envi_file_context()`"
        )
    if bbl == "No BBL Found":
        bbl = [1] * len(wvls)

    if prf["crs"] is None:
        crs = GeographicCRS.GCS_MOON_2000
    else:
        crs = prf["crs"]

    context_dict: ContextBuilder = {
        "name": name,
        "description": desc,
        "data_filename": Path(Path(fp).name),
        "nrows": prf["height"],
        "ncols": prf["width"],
        "nbands": prf["count"],
        "crs": crs,
        "geotransform": GeotransformModel.fromaffine(prf["transform"]),
        "hdr_off": 0,
        "data_type": NumpyDType.FLOAT32,
        "interleave": "BIL",
        "nodata": -999,
        "band_names": [f"Band{n+1}({i}nm)" for n, i in enumerate(wvls)],
        "measurement_name": "Wavelength",
        "measurement_units": "nm",
        "measurement_values": wvls,
        "bad_bands": bbl,
        "id": uuid4(),
    }

    return CubeContext.from_builder(context_dict)


def read_measurement_envi_file_context(
    fp: Path | str, name: str
) -> CubeContext:
    """
    Reads the context data from an envi file that does not represent a spectral
    dataset, but some other measurement (Location, Geometry, Parameters,
    etc...)

    Parameters
    ----------
    fp : Path | str
        File path to the ENVI cube.
    name : str
        Name of the CubeContext object.

    Returns
    -------
    CubeContext
        Resulting context data from the cube.
    """

    with rio.open(fp, "r") as f:
        prf: RasterioProfile = f.profile

    hdr_fp = Path(fp).with_suffix(".hdr")
    band_names = extract_hdr_band_names(hdr_fp)
    bbl = extract_hdr_bbl(hdr_fp)
    desc = extract_hdr_desc(hdr_fp)
    if bbl == "No BBL Found":
        bbl = [1] * len(band_names)

    if prf["crs"] is None:
        crs = GeographicCRS.GCS_MOON_2000
    else:
        crs = prf["crs"]

    context_dict: ContextBuilder = {
        "name": name,
        "description": desc,
        "data_filename": Path(Path(fp).name),
        "nrows": prf["height"],
        "ncols": prf["width"],
        "nbands": prf["count"],
        "crs": crs,
        "geotransform": GeotransformModel.fromaffine(prf["transform"]),
        "hdr_off": 0,
        "data_type": NumpyDType(prf["dtype"]),
        "interleave": "BIL",
        "nodata": -999,
        "band_names": band_names,
        "measurement_name": name,
        "measurement_units": "unitless",
        "measurement_values": [i for i in range(prf["count"])],
        "bad_bands": bbl,
        "id": uuid4(),
    }

    return CubeContext.from_builder(context_dict)
