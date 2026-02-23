# Dependencies
import numpy as np
from scipy.spatial import Delaunay
import xarray as xr

# Local
from .models.gcp_model import GCPGroup


def generate_latlong(
    col_gcps: np.ndarray,
    row_gcps: np.ndarray,
    lon_gcps: np.ndarray,
    lat_gcps: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    # GCP arrays
    pix = np.column_stack([col_gcps, row_gcps])
    latlon = np.column_stack([lat_gcps, lon_gcps])

    tri = Delaunay(pix)

    # Full pixel grid
    jj, ii = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pts = np.column_stack([ii.ravel(), jj.ravel()])

    simplices = tri.find_simplex(pts)
    T = tri.transform[simplices]

    bary = np.einsum("ijk,ik->ij", T[:, :2, :], pts - T[:, 2])
    bary = np.c_[bary, 1 - bary.sum(axis=1)]

    latlon_dense: np.ndarray = np.sum(
        latlon[tri.simplices[simplices]] * bary[..., None], axis=1
    )

    return latlon_dense.reshape((height, width, 2))


def latlong_from_gcp_group(
    gcp_group: GCPGroup, base_image: np.ndarray | xr.DataArray
) -> np.ndarray:
    return generate_latlong(
        gcp_group.col_pixels,
        gcp_group.row_pixels,
        gcp_group.map_x,
        gcp_group.map_y,
        base_image.shape[0],
        base_image.shape[1],
    )
