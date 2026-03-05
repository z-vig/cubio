"""
Utilities for taking data that is already on a regular geospatial coordinate
grid and resasmpling it to another
"""

# Dependencies
from pyresample.geometry import AreaDefinition
from pyresample.kd_tree import resample_nearest
from pyproj.crs import CRS
from pyproj.transformer import Transformer
import numpy as np
import xarray as xr

# Local
from cubio.geotools.models import GeotransformModel, BoundingBoxModel


def resample_regular_grid_array(
    regular_grid_array: np.ndarray,
    src_crs: CRS,
    src_geotransform: GeotransformModel,
    trg_crs: CRS,
    trg_array_size: tuple[int, int],
) -> xr.DataArray:
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
        xs.max(),
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
        trg_crs.srs,
        trg_array_size[0],
        trg_array_size[1],
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

    return xr.DataArray(resamp)
