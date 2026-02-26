# Built-Ins
from pathlib import Path
from copy import copy

# Dependencies
import numpy as np
import xarray as xr

# Local
from .models import BoundingBoxModel, GCPGroup
from .georeference_satellite_swath import (
    georeference_satellite_swath,
    ProjectionDefinition,
)
from .generate_geoloc_backplane import latlong_from_gcp_group
from cubio.cube_context import CubeContext
from cubio.cube_data import CubeData
from cubio.geotools.models import ImageOffset, GeotransformModel


def georeference_image(
    cubio_json_file: Path | str,
    gcps_file: Path | str,
    prj_definition: ProjectionDefinition,
    unref_cube_array: np.ndarray | None = None,
    georef_extent: BoundingBoxModel | None = None,
    new_gcps_offset: ImageOffset | None = None,
    apply_cropping: bool = True,
) -> tuple[np.ndarray, GeotransformModel]:
    # ---- Reading Image Context and Lazy Loading ----
    unreferenced_context = CubeContext.from_json(cubio_json_file)
    if unref_cube_array is None:
        unref_cube = unreferenced_context.lazy_load_data()
        unref_cube.transpose_to("BIP")
    elif unref_cube_array is not None:
        unref_cube = CubeData("unref_cube", format="BIP")
        unref_cube.array = xr.DataArray(unref_cube_array)

    # ---- Cropping to extent of GCP group ----
    gcp_group = GCPGroup.from_gcps_file(gcps_file)
    if new_gcps_offset:
        gcp_group.adjust_offset(new_gcps_offset)
    if apply_cropping:
        offset_cube: np.ndarray = np.array(
            gcp_group.offset.crop_image(unref_cube.array)
        )
    else:
        offset_cube = np.array(unref_cube.array)

    # ---- Creating Lat/Long Backplane ----
    latlongarr = latlong_from_gcp_group(gcp_group, offset_cube)
    long = latlongarr[:, :, 1]
    lat = latlongarr[:, :, 0]

    # ---- Automatically obtaining extent, if one is not provided ----
    if georef_extent is None:
        ext = BoundingBoxModel(
            name="auto",
            top=lat.max() - 0.2,
            bottom=lat.min() + 0.2,
            left=long.min() + 0.2,
            right=long.max() - 0.2,
        )

    else:
        ext = georef_extent
    # ---- Resampling Image ----
    resamp_img, gtrans = georeference_satellite_swath(
        satellite_data=offset_cube,
        longitude_backplane=long,
        latitude_backplane=lat,
        proj=prj_definition,
        extent=ext,
    )

    return resamp_img, gtrans


def save_georeference(
    cubio_json_file: Path | str,
    resamp_img: np.ndarray,
    gtrans: GeotransformModel,
    prj_definition: ProjectionDefinition,
    unreferenced_context: CubeContext,
    save_directory: Path | str | None = None,
    new_name: str | None = None,
    new_description: str | None = None,
    save_new_context: bool = True,
):
    # ---- Export Main Array to Disk ----
    xcoord, ycoord = gtrans.generate_coords(
        height=resamp_img.shape[0], width=resamp_img.shape[1]
    )
    meas_name = unreferenced_context.measurement_name
    meas_vals = unreferenced_context.measurement_values
    darray = xr.DataArray(
        resamp_img,
        coords={
            "Latitude": ycoord,
            "Longitude": xcoord,
            meas_name: meas_vals,
        },
        dims=("Latitude", "Longitude", meas_name),
    )
    if save_directory is None:
        src_fp = Path(cubio_json_file)
        dst_fp = src_fp.with_name(f"{src_fp.stem}_georef")
        save_fp = dst_fp.with_suffix(".zarr")
    else:
        dst_fp = Path(save_directory, unreferenced_context.name)
        save_fp = dst_fp.with_suffix(".zarr")
    darray.to_zarr(save_fp, mode="w", consolidated=True, zarr_format=2)

    referenced_context = copy(unreferenced_context.builder)
    referenced_context.update(
        {
            "crs": prj_definition.crs_wkt_str,
            "geotransform": gtrans,
            "nrows": resamp_img.shape[0],
            "ncols": resamp_img.shape[1],
            "data_filename": Path(save_fp.stem),
        }
    )

    if new_name is not None:
        referenced_context["name"] = new_name
    if new_description is not None:
        referenced_context["description"] = new_description

    georef_context = CubeContext.from_builder(referenced_context)

    if save_new_context:
        georef_context.write_json(save_fp.with_suffix(".json"))

    return georef_context
