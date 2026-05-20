# Built-Ins
from pathlib import Path
from typing import Literal

# Dependencies
from shapely.geometry import Polygon
import shapely
import xarray as xr
import numpy as np
import geopandas as gpd

# Local
from cubio.types import LabelLike


def open_shapefile_as_shapely_polygon(
    shapefile_fp: str | Path, handle_geoms: Literal["first", "all"] = "all"
) -> Polygon | list[Polygon]:
    """
    Opens a shapefile and returns a shapely Polygon object.

    Parameters
    ----------
    shapefile_fp: str | Path
        Path to shapefile.
    """
    gdf = gpd.read_file(shapefile_fp)
    if handle_geoms == "first":
        poly = Polygon(gdf["geomety"].iloc[0])
        return poly
    elif handle_geoms == "all":
        poly_list: list[Polygon] = []
        for _, row in gdf.iterrows():
            poly_list.append(row["geometry"])
        return poly_list
    else:
        raise ValueError("Invalid handle_geoms arg.")


def raster_from_polygon_list(
    lat_index: tuple[str, LabelLike],
    lon_index: tuple[str, LabelLike],
    polygon_list: list[Polygon],
) -> xr.DataArray:
    lat_dense, lon_dense = np.meshgrid(lat_index[1], lon_index[1])
    points = shapely.points(lon_dense, lat_dense)
    mask_list: list[xr.DataArray] = []
    shapefile_overlap = False
    for polygon in polygon_list:
        poly_raster = xr.DataArray(
            shapely.contains(polygon, points).T,
            coords={lat_index[0]: lat_index[1], lon_index[0]: lon_index[1]},
            dims=(lat_index[0], lon_index[0]),
        )

        if np.all(poly_raster == 0):
            continue

        mask_list.append(poly_raster)
        shapefile_overlap = True

    if not shapefile_overlap:
        raise ValueError("Input shapefile does not overlap with Cube Data.")

    full_poly_raster = xr.DataArray(
        np.zeros_like(lat_dense, dtype=bool),
        coords={lat_index[0]: lat_index[1], lon_index[0]: lon_index[1]},
        dims=(lon_index[0], lat_index[0]),
    )
    for i in mask_list:
        full_poly_raster = full_poly_raster | i

    return full_poly_raster


def raster_from_single_polygon(
    lat_index: tuple[str, LabelLike],
    lon_index: tuple[str, LabelLike],
    polygon: Polygon,
) -> xr.DataArray:
    """
    Creates a raster that highlights the location of a shapely Polygon object
    on a raster array, given a uniformly sampled lat/long grid.

    Parameters
    ----------
    lat_index: tuple[str, LabelLike]
        1-D Latitude index of the uniform geolocation array.
    lon_index: tuple[str, LabelLike]
        1-D Longitude index of the uniform geolocation array.
    polygon: Polygon
        Shapely polygon object.

    Notes
    -----
    Because the lat/long grid must be uniformly spaced, only two 1-D arrays
    are required to create the entire lat/long grid.
    """
    lat_dense, lon_dense = np.meshgrid(lat_index[1], lon_index[1])
    points = shapely.points(lon_dense, lat_dense)
    poly_raster = xr.DataArray(
        shapely.contains(polygon, points).T,
        coords={lat_index[0]: lat_index[1], lon_index[0]: lon_index[1]},
        dims=(lat_index[0], lon_index[0]),
    )
    if poly_raster is False:
        raise ValueError(
            "The polygon lies outside of the provided lat/long grid."
        )
    return poly_raster


def raster_from_shapefile(
    lat_coord: tuple[str, LabelLike],
    lon_coord: tuple[str, LabelLike],
    shapefile_fp: str | Path,
) -> xr.DataArray:
    """
    Generates a boolean raster from a uniform lat/long backplane.

    Parameters
    ----------
    lat_index: tuple[str, LabelLike]
        1-D Latitude index of the uniform geolocation array, with its name.
    lon_index: tuple[str, LabelLike]
        1-D Longitude index of the uniform geolocation array, with its name.
    shapefile_fp: str | Path
        Path to shapefile.
    """
    poly = open_shapefile_as_shapely_polygon(shapefile_fp)
    if isinstance(poly, Polygon):
        arr = raster_from_single_polygon(lat_coord, lon_coord, poly)
    elif isinstance(poly, list):
        arr = raster_from_polygon_list(lat_coord, lon_coord, poly)
    else:
        raise ValueError("Invalid polygon.")
    return arr
