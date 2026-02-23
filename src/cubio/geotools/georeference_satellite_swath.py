# Built-Ins
from dataclasses import dataclass
from typing_extensions import Self

# Dependencies
import numpy as np
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample import kd_tree
import xarray as xr
from tqdm import tqdm

# Local
from .models import BoundingBoxModel
from cubio.geotools.models import GeotransformModel, PointModel

MOON_RADIUS = 1737400
MOON_M_PER_DEG = np.pi * MOON_RADIUS / 180


@dataclass
class ProjectionDefinition:
    area_name: str
    proj_name: str
    description: str
    proj4_str: str
    crs_wkt_str: str


@dataclass
class PixelResolution:
    pix_per_deg: float
    deg_per_pix: float
    m_per_pix: float  # Estimation

    @classmethod
    def from_array(cls, max_lat: float, min_lat: float, height: int) -> Self:
        lat_height = max_lat - min_lat
        dpp: float = lat_height / height  # Degrees per pixel
        return cls(
            pix_per_deg=1 / dpp,
            deg_per_pix=dpp,
            m_per_pix=MOON_M_PER_DEG * dpp,
        )


def georeference_satellite_swath(
    satellite_data: np.ndarray | xr.DataArray,
    longitude_backplane: np.ndarray | xr.DataArray,
    latitude_backplane: np.ndarray | xr.DataArray,
    proj: ProjectionDefinition,
    extent: BoundingBoxModel,
) -> tuple[np.ndarray, GeotransformModel]:
    # ---- Automatically detecting resolution ----
    max_lat = float(latitude_backplane[0, :].max())
    min_lat = float(latitude_backplane[-1, :].min())
    res = PixelResolution.from_array(
        max_lat=max_lat, min_lat=min_lat, height=satellite_data.shape[0]
    )

    # ---- Defining Swath ----
    swath = SwathDefinition(lons=longitude_backplane, lats=latitude_backplane)

    # ---- Defining Projection Area ---
    area_height = (extent.top - extent.bottom) * res.pix_per_deg
    area_width = (extent.right - extent.left) * res.pix_per_deg

    area = AreaDefinition(
        proj.area_name,
        proj.description,
        proj.proj_name,
        proj.proj4_str,
        area_width,
        area_height,
        area_extent=extent.as_extent(mode="BottomLeft"),
    )

    # ---- Resampling ----
    resampled_data: np.ndarray
    if satellite_data.ndim == 2:
        resampled_data = kd_tree.resample_nearest(
            swath,
            satellite_data,
            area,
            res.m_per_pix * 4,
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )
    elif satellite_data.ndim == 3:
        first_band_resamp = kd_tree.resample_nearest(
            swath,
            satellite_data[:, :, 0],
            area,
            res.m_per_pix * 3,
            epsilon=0.5,  # type: ignore
            fill_value=np.nan,  # type: ignore
        )
        if not isinstance(first_band_resamp, np.ndarray):
            raise ValueError("Invalid first band resampling.")
        resampled_data = np.empty(
            (*first_band_resamp.shape, satellite_data.shape[2])
        )
        resampled_data[:, :, 0] = first_band_resamp
        for band in tqdm(
            range(1, satellite_data.shape[2]),
            desc="Resampling bands...",
            total=satellite_data.shape[2] - 1,
        ):
            resampled_data[:, :, band] = kd_tree.resample_nearest(
                swath,
                satellite_data[:, :, band],
                area,
                res.m_per_pix * 3,
                epsilon=0.5,  # type: ignore
                fill_value=np.nan,  # type: ignore
            )
    else:
        raise ValueError(
            "Satellite data has an invalid number of dimensions:"
            f" {satellite_data.ndim}."
        )

    gtrans = GeotransformModel(
        upperleft=PointModel(x=extent.top_left.x, y=extent.top_left.y),
        xres=res.deg_per_pix,
        yres=-res.deg_per_pix,
        col_rotation=0,
        row_rotation=0,
    )

    return resampled_data, gtrans
