from typing import Optional

from cubio.cube_data.cube_data import CubeData
from cubio.cube_context.cube_context import CubeContext
from cubio.geotools.models.bounding_box_model import bbox_intersection
from cubio.geotools.models import BoundingBoxModel


def _bbox_in_cubedata(cubedata: CubeData, bbox: BoundingBoxModel) -> bool:
    data_bounds = cubedata.bounds
    intersection = bbox_intersection(data_bounds, bbox)
    if intersection == "No Intersection Found.":
        return False
    return True


def crop_cubedata(
    cubecontext: CubeContext,
    cubedata: CubeData,
    bbox: BoundingBoxModel,
    *,
    name: Optional[str] = None,
    desc: Optional[str] = None,
    new_filename: Optional[str] = None,
) -> tuple[CubeContext, CubeData]:
    if not _bbox_in_cubedata(cubedata, bbox):
        raise ValueError("Data and bounding box do not overlap for cropping.")

    cropped_data, cropped_gtrans = cubedata.read_bbox(bbox)

    if name is None:
        name = f"{cubedata.name}_cropped"
    if desc is None:
        desc = (
            f"CubeData named {cubedata.name} cropped to a user-specified "
            "bounding box."
        )
    if new_filename is None:
        new_filename = f"{cubecontext.data_filename}_cropped"

    cropped_bldr = cubecontext.builder
    cropped_bldr.update(
        {
            "name": name,
            "description": desc,
            "data_filename": new_filename,
            "geotransform": cropped_gtrans,
            "nrows": cropped_data.shape[0],
            "ncols": cropped_data.shape[1],
        }
    )
    cropped_cc = CubeContext.from_builder(cropped_bldr)

    cropped_cb = CubeData(name, cubedata.fmt)
    cropped_cb.array = cropped_data

    return cropped_cc, cropped_cb
