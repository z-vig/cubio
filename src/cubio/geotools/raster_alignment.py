"""
Functions for aligning one raster dataset to the pixel grid of another.
"""

# Built-Ins
from typing import Literal
from pathlib import Path

# Dependencies
import numpy as np
from pyproj.crs import CRS
from pyresample.geometry import AreaDefinition
from pyresample.kd_tree import resample_nearest
import xarray as xr

from cubio.geotools.models import GeotransformModel, BoundingBoxModel
from cubio.cube_data import CubeData
from cubio.cube_context import CubeContext


def align_raster_data(
    source_raster: np.ndarray,
    source_crs: CRS,
    source_geotransform: GeotransformModel,
    grid_raster: np.ndarray,
    grid_crs: CRS,
    grid_geotransform: GeotransformModel,
):
    MOON_RADIUS = 1737400
    MOON_M_PER_DEG = np.pi * MOON_RADIUS / 180

    src_bbox = source_geotransform.get_bbox(
        source_raster.shape[0], source_raster.shape[1]
    )
    trg_bbox = grid_geotransform.get_bbox(
        grid_raster.shape[0], grid_raster.shape[1]
    )

    src_area = AreaDefinition(
        "src_area",
        "Source Area",
        "src_crs",
        source_crs,
        width=source_raster.shape[1],
        height=source_raster.shape[0],
        area_extent=src_bbox.as_extent(),
    )

    trg_area = AreaDefinition(
        "trg_area",
        "Target Area",
        "trg_crs",
        grid_crs,
        width=grid_raster.shape[1],
        height=grid_raster.shape[0],
        area_extent=trg_bbox.as_extent(),
    )

    roi = source_geotransform.xres * MOON_M_PER_DEG

    aligned_source: np.ndarray = resample_nearest(
        src_area,
        source_raster,
        trg_area,
        roi * 3,
        epsilon=0.5,  # type: ignore
        fill_value=np.nan,  # type: ignore
    )

    aligned_gtrans = GeotransformModel.fromarraysize(
        upper_left_latitiude=trg_bbox.top,
        upper_left_longitude=trg_bbox.left,
        lower_right_latitude=trg_bbox.bottom,
        lower_right_longitude=trg_bbox.right,
        height=aligned_source.shape[0],
        width=aligned_source.shape[1],
    )

    return xr.DataArray(aligned_source), aligned_gtrans


def align_datacubes(
    src_cubecontext: CubeContext,
    src_cubedata: CubeData,
    trg_cubecontext: CubeContext,
    trg_cubedata: CubeData,
    *,
    src_bbox: BoundingBoxModel | Literal["FullArray"] = "FullArray",
    trg_bbox: BoundingBoxModel | Literal["FullArray"] = "FullArray",
    new_filename: Path | None = None,
) -> tuple[CubeContext, CubeData]:
    src_cubedata.transpose_to("BIP")
    trg_cubedata.transpose_to("BIP")

    if isinstance(src_bbox, BoundingBoxModel):
        src_arr, src_gtrans = src_cubedata.read_bbox(src_bbox)
    else:
        src_arr = src_cubedata.array
        src_gtrans = src_cubecontext.geotransform

    if isinstance(trg_bbox, BoundingBoxModel):
        trg_arr, trg_gtrans = trg_cubedata.read_bbox(trg_bbox)
    else:
        trg_arr = trg_cubedata.array
        trg_gtrans = trg_cubecontext.geotransform

    aligned_array, aligned_gtrans = align_raster_data(
        np.array(src_arr),
        CRS.from_string(src_cubecontext.crs),
        src_gtrans,
        np.array(trg_arr),
        CRS.from_string(trg_cubecontext.crs),
        trg_gtrans,
    )

    if new_filename is None:
        align_fp = src_cubecontext.data_filename
    else:
        align_fp = new_filename
    align_builder = src_cubecontext.builder
    align_builder.update(
        {
            "data_filename": align_fp,
            "crs": trg_cubecontext.crs,
            "geotransform": aligned_gtrans,
            "nrows": aligned_array.shape[0],
            "ncols": aligned_array.shape[1],
        }
    )

    aligned_cubedata = CubeData(
        f"{src_cubedata.name}_{trg_cubedata.name}_alignment", "BIP"
    )
    aligned_cubedata.array = aligned_array

    return (CubeContext.from_builder(align_builder), aligned_cubedata)
