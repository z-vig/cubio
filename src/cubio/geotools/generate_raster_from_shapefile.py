# Built-Ins
from pathlib import Path

# Dependencies
from shapely.geometry import Polygon
import shapely
import xarray as xr
import numpy as np
import geopandas as gpd

# Local
from cubio.types import LabelLike


def open_shapefile_as_shapely_polygon(shapefile_fp: str | Path) -> Polygon:
    """
    Opens a shapefile and returns a shapely Polygon object.

    Parameters
    ----------
    shapefile_fp: str | Path
        Path to shapefile.
    """
    gdf = gpd.read_file(shapefile_fp)
    poly = Polygon(gdf["geometry"].iloc[0])
    return poly


def raster_from_shapely_polygon(
    lat_index: LabelLike, lon_index: LabelLike, polygon: Polygon
) -> xr.DataArray:
    """
    Creates a raster that highlights the location of a shapely Polygon object
    on a raster array, given a uniformly sampled lat/long grid.

    Parameters
    ----------
    lat_index: LabelLike
        1-D Latitude index of the uniform geolocation array.
    lon_index: LabelLike
        1-D Longitude index of the uniform geolocation array.
    polygon: Polygon
        Shapely polygon object.

    Notes
    -----
    Because the lat/long grid must be uniformly spaced, only two 1-D arrays
    are required to create the entire lat/long grid.
    """
    lat_dense, lon_dense = np.meshgrid(lat_index, lon_index)
    points = shapely.points(lon_dense, lat_dense)
    poly_raster = xr.DataArray(
        shapely.contains(polygon, points).T,
        coords={"Latitude": lat_index, "Longitude": lon_index},
        dims=("Latitude", "Longitude"),
    )
    if poly_raster is False:
        raise ValueError(
            "The polygon lies outside of the provided lat/long grid."
        )
    return poly_raster


def raster_from_shapefile(
    lat_index: LabelLike, lon_index: LabelLike, shapefile_fp: str | Path
) -> xr.DataArray:
    """
    Generates a boolean raster from a uniform lat/long backplane.

    Parameters
    ----------
    lat_index: LabelLike
        1-D Latitude index of the uniform geolocation array.
    lon_index: LabelLike
        1-D Longitude index of the uniform geolocation array.
    shapefile_fp: str | Path
        Path to shapefile.
    """
    poly = open_shapefile_as_shapely_polygon(shapefile_fp)
    arr = raster_from_shapely_polygon(lat_index, lon_index, poly)
    return arr
