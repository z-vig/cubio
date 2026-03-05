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
    lat_index: LabelLike, lon_index: LabelLike, polygon_list: list[Polygon]
) -> xr.DataArray:
    print("LIST PROCESSING")
    lat_dense, lon_dense = np.meshgrid(lat_index, lon_index)
    points = shapely.points(lon_dense, lat_dense)
    mask_list: list[xr.DataArray] = []
    for polygon in polygon_list:
        poly_raster = xr.DataArray(
            shapely.contains(polygon, points).T,
            coords={"Latitude": lat_index, "Longitude": lon_index},
            dims=("Latitude", "Longitude"),
        )

        if poly_raster is False:
            raise ValueError(
                "The polygon lies outside of the provided lat/long grid."
            )

        mask_list.append(poly_raster)

        print(poly_raster.coords)
    full_poly_raster = xr.DataArray(
        np.zeros_like(lat_dense, dtype=bool),
        coords={"Latitude": lat_index, "Longitude": lon_index},
        dims=("Longitude", "Latitude"),
    )
    for i in mask_list:
        full_poly_raster = full_poly_raster | i

    return full_poly_raster


def raster_from_single_polygon(
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
    if isinstance(poly, Polygon):
        arr = raster_from_single_polygon(lat_index, lon_index, poly)
    elif isinstance(poly, list):
        arr = raster_from_polygon_list(lat_index, lon_index, poly)
    else:
        raise ValueError("Invalid polygon.")
    return arr
