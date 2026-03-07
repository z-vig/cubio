"""
Utilities for taking data that is already on a regular geospatial coordinate
grid and resasmpling it to another
"""

# Built-Ins
from typing import Literal
from pathlib import Path
import os

# Dependencies
from pyresample.geometry import AreaDefinition
from pyresample.kd_tree import resample_nearest
from pyproj.crs import CRS
from pyproj.transformer import Transformer
import numpy as np
import xarray as xr

# Local
from cubio.geotools.models import (
    GeotransformModel,
    BoundingBoxModel,
    PointModel,
)
from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData


def resample_regular_grid_array(
    regular_grid_array: np.ndarray,
    src_crs: CRS,
    src_geotransform: GeotransformModel,
    trg_crs: CRS,
    trg_array_size: tuple[int, int],
    north_up: bool = True,
) -> tuple[xr.DataArray, GeotransformModel]:
    """
    Resample a regular grid array to another regular grid defined by the target
    CRS and array size.

    Parameters
    ----------
    regular_grid_array: np.ndarray
        The input array on a regular grid.
    src_crs: CRS
        The coordinate reference system of the input array.
    src_geotransform: GeotransformModel
        The geotransform of the input array.
    trg_crs: CRS
        The coordinate reference system of the output array.
    trg_array_size: tuple[int, int]
        The size of the output array (height, width).
    """
    MOON_RADIUS = 1737400
    MOON_M_PER_DEG = np.pi * MOON_RADIUS / 180
    transformer = Transformer.from_crs(src_crs, trg_crs, always_xy=True)

    src_bbox: BoundingBoxModel = src_geotransform.get_bbox(
        regular_grid_array.shape[0], regular_grid_array.shape[1]
    )

    xs, ys = transformer.transform(
        [
            src_bbox.bottom_left.x,
            src_bbox.bottom_right.x,
            src_bbox.top_left.x,
            src_bbox.top_right.x,
        ],
        [
            src_bbox.bottom_left.y,
            src_bbox.bottom_right.y,
            src_bbox.top_left.y,
            src_bbox.top_right.y,
        ],
    )
    xs = np.array(xs)
    ys = np.array(ys)

    trg_area_extent = (
        xs.min(),
        ys.min(),
        xs.max(),
        ys.max(),
    )

    src_area = AreaDefinition(
        "src",
        "Source Area",
        "src",
        src_crs,
        regular_grid_array.shape[1],
        regular_grid_array.shape[0],
        area_extent=src_bbox.as_extent(),
    )

    trg_area = AreaDefinition(
        "trg",
        "Target Area",
        "trg",
        trg_crs,
        trg_array_size[1],
        trg_array_size[0],
        area_extent=trg_area_extent,
    )

    roi = src_geotransform.xres * MOON_M_PER_DEG

    resamp = resample_nearest(
        src_area,
        regular_grid_array,
        trg_area,
        roi * 3,
        epsilon=0.5,  # type: ignore
        fill_value=np.nan,  # type: ignore
    )
    xresolution = (trg_area_extent[2] - trg_area_extent[0]) / trg_array_size[1]
    yresolution = (trg_area_extent[3] - trg_area_extent[1]) / trg_array_size[0]
    if north_up:
        yresolution *= -1

    resamp_geotransform = GeotransformModel(
        upperleft=PointModel(x=trg_area_extent[0], y=trg_area_extent[3]),
        xres=xresolution,
        yres=yresolution,
        row_rotation=0,
        col_rotation=0,
    )

    return xr.DataArray(resamp), resamp_geotransform


def resample_regular_cubedata(
    src_cubedata: CubeData,
    src_cubecontext: CubeContext,
    src_bbox: BoundingBoxModel | Literal["FullArray"],
    trg_crs: CRS,
    *,
    trg_array_size: tuple[int, int] | None = None,
    new_filename: Path | None = None,
    ignore_celestial_body: bool = False,
) -> tuple[CubeContext, CubeData]:
    if ignore_celestial_body:
        os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "YES"

    if src_bbox == "FullArray":
        arr = src_cubedata.array
        src_gtrans = src_cubecontext.geotransform
    else:
        arr, src_gtrans = src_cubedata.read_bbox(src_bbox)

    if trg_array_size is None:
        trg_size = (arr.shape[0], arr.shape[1])
    else:
        trg_size = trg_array_size

    resamp, resamp_gtrans = resample_regular_grid_array(
        np.array(arr),
        CRS.from_string(src_cubecontext.crs),
        src_gtrans,
        trg_crs,
        trg_size,
    )

    if new_filename is None:
        resamp_fp = src_cubecontext.data_filename
    else:
        resamp_fp = new_filename
    resamp_builder = src_cubecontext.builder
    resamp_builder.update(
        {
            "data_filename": resamp_fp,
            "crs": trg_crs.to_string(),
            "geotransform": resamp_gtrans,
        }
    )

    resamp_cubedata = CubeData(src_cubedata.name, src_cubedata.fmt)
    resamp_cubedata.array = resamp

    return CubeContext.from_builder(resamp_builder), resamp_cubedata
